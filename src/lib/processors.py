import logging
import sys
import typing as t

import nptyping as npt
import numpy as np

import tdt

from . import structures, utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Offset in the filtered["time ranges"] for start and end of the time ranges included
#  via a filtering search
# i.e. filtered = tdt.epoc_filter(data, "TriS", t=[-10, 20])
#  would have filtered["time_ranges"]
TIME_RANGE_ONSET_IDX = 0
TIME_RANGE_OFFSET_IDX = 1

# Time (in seconds) that an epoch timestamp can fall outside of a 'stimulus' epoch and still be considered
#  inside that stimulus (e.g. stimulus was at 131.012, but acoustic frequency at 131.008; if this difference
#  is less than the margin then they will be treated as simultaneous)
# In practice this seems to be around .0001 between the acoustic frequency and stim with stim trailing
EPOCH_TIMESTAMP_ERROR_MARGIN = 0.001


class SessionProcessor:
    """
    Process a acclim/training session for Simeon's 2009 research
    """

    tone_duration: float = 0.25
    inter_tone_interval: float = 0.25
    in_tone_capture_start_offset: float = 0.0
    in_tone_capture_end_offset: float = 0.2
    out_tone_capture_start_offset: float = 0.1
    out_tone_capture_end_offset: float = 0.25

    trial_epoch = "TriS"
    stimulus_epoch = "StiS"

    # acoustic parameters
    attenuation_epoch = "Attn"
    acoustic_frequency_epoch = "AFrq"

    # neural stim parameters
    reference_channels_bitmask_epoch = "ReBM"
    stimulation_channels_bitmask_epoch = "StBM"
    stimulation_current_epoch = "Curr"
    stimulation_frequency_epoch = "Freq"

    # Epochs of unknown meaning
    # block_start_epoch = "BloS"  # Not always present? Has no values?
    # unknown_1_epoch = "SweS"

    # Cache for filtered data to trials - mainly for development to avoid having to refilter a lot
    _block_data: t.Any = None
    _trial_windows: t.Any = None

    def __init__(self, *, block_path: str):
        self.block_path = block_path

    def get_parameter_summary(self) -> str:
        """Return a summary of the current configuration of the Session Processor"""
        summary_fields: t.Dict[str, t.List[str]] = {
            "Tones": ["tone_duration", "inter_tone_interval"],
            "Offsets": [
                "in_tone_capture_start_offset",
                "in_tone_capture_end_offset",
                "out_tone_capture_start_offset",
                "out_tone_capture_end_offset",
            ],
            "Epochs epochs": [
                "trial_epoch",
                "stimulus_epoch",
                "attenuation_epoch",
                "acoustic_frequency_epoch",
                "reference_channels_bitmask_epoch",
                "stimulation_channels_bitmask_epoch",
                "stimulation_current_epoch",
                "stimulation_frequency_epoch",
            ],
            "Block": ["block_path"],
        }

        output: t.List[str] = []
        for heading, fields in summary_fields.items():
            if output:
                output.append("\n")
            output.append(heading.upper())
            output.append("-" * len(heading))
            for field in fields:
                output.append(f"{field.replace('_', ' ').title()}: {getattr(self, field)}")

        return "\n".join(output)

    def set_tone_parameters(self, *, tone_duration: float, inter_tone_interval: float) -> None:
        """Set duration and interval of the presented tones

        Parameters
        ----------
        tone_duration :
            Duration of tones presented in seconds
        inter_tone_interval :
            Inter-tone interval in seconds
        """
        self.tone_duration = tone_duration
        self.inter_tone_interval = inter_tone_interval

    def set_extraction_parameters(
        self,
        *,
        in_tone_capture_start_offset: float,
        in_tone_capture_end_offset: float,
        out_tone_capture_start_offset: float,
        out_tone_capture_end_offset: float,
    ) -> None:
        """Set time windows for spike count inclusion during extraction

        Parameters
        ----------
        in_tone_capture_start_offset :
            Time after tone start from which spikes will be included in the count
        in_tone_capture_end_offset :
            Time after tone start until which spikes will be included in the count
        out_tone_capture_start_offset :
            Time after inter-tone interval start from which spikes will be included in the count
        out_tone_capture_end_offset :
            Time after inter-tone interval start until which spikes will be included in the count
        """
        self.in_tone_capture_start_offset = in_tone_capture_start_offset
        self.in_tone_capture_end_offset = in_tone_capture_end_offset
        self.out_tone_capture_start_offset = out_tone_capture_start_offset
        self.out_tone_capture_end_offset = out_tone_capture_end_offset

    def set_epoch_names(
        self,
        *,
        trial_epoch: t.Optional[str] = None,
        stimulus_epoch: t.Optional[str] = None,
        attenuation_epoch: t.Optional[str] = None,
        acoustic_frequency_epoch: t.Optional[str] = None,
        reference_channels_bitmask_epoch: t.Optional[str] = None,
        stimulation_channels_bitmask_epoch: t.Optional[str] = None,
        stimulation_current_epoch: t.Optional[str] = None,
        stimulation_frequency_epoch: t.Optional[str] = None,
    ) -> None:
        """Replace the epoch name for one or more session epoch. Any epochs not specified
            are not replaced.

        Parameters
        ----------
        trial_epoch :
            Name of epoch representing trial count and duration
        stimulus_epoch :
            Name of epoch representing stimulus count and duration
        attenuation_epoch :
            Name of epoch containing acoustic tone attenuation
        acoustic_frequency_epoch :
            Name of epoch containing acoustic tone frequency
        reference_channels_bitmask_epoch :
            Name of epoch containing neural reference channels bitmask
        stimulation_channels_bitmask_epoch :
            Name of epoch containing neural stimulation channels bitmask
        stimulation_current_epoch :
            Name of epoch containing neural stimulation current
        stimulation_frequency_epoch :
            Name of epoch containing neural stimulation frequency
        """
        if trial_epoch and trial_epoch != self.trial_epoch:
            self.trial_epoch = trial_epoch
            self.trial_windows = None

        self.stimulus_epoch = stimulus_epoch or self.stimulus_epoch
        self.attenuation_epoch = attenuation_epoch or self.attenuation_epoch
        self.acoustic_frequency_epoch = acoustic_frequency_epoch or self.acoustic_frequency_epoch
        self.reference_channels_bitmask_epoch = (
            reference_channels_bitmask_epoch or self.reference_channels_bitmask_epoch
        )
        self.stimulation_channels_bitmask_epoch = (
            stimulation_channels_bitmask_epoch or self.stimulation_channels_bitmask_epoch
        )
        self.stimulation_current_epoch = stimulation_current_epoch or self.stimulation_current_epoch

    def extract_spikes_from_trial(self, *, from_trial_offset: float, to_trial_offset: float) -> structures.Session:
        """
        Extracts spike counts for each tone of each trial in the session

        Parameters
        ----------
        from_trial_offset :
            Offset from trial start from which to include spike data per tone (negative for before trial start)
        to_trial_offset :
            Offset from trial start until which to include spike data per tone
            (note: any incomplete tones at the end are not included)

        Returns
        -------
        Session object for the testing session

        """
        tones_per_second: float = 1 / (self.tone_duration + self.inter_tone_interval)
        if not self._block_data:
            self._block_data = tdt.read_block(self.block_path, evtype=["epocs", "snips", "scalars"], nodata=1)
        if not self._trial_windows:
            with utils.timer(label="TDT epoc_filter", logger=logger):
                self._trial_windows = tdt.epoc_filter(
                    self._block_data, "TriS", t=[from_trial_offset, to_trial_offset - from_trial_offset]
                )
        trial_windows = self._trial_windows

        # Skip the last tone as the end offset should always go a little past the end
        # tone_count_to_include = int(((end_offset - start_offset) * tones_per_second) - 1)

        cspk_offset = 0
        cspk_data = trial_windows["snips"]["CSPK"]
        cspk_length = len(cspk_data["ts"])
        np.set_printoptions(threshold=sys.maxsize)

        stimulus_epoch_idx = 0
        attenuation_epoch_idx = 0
        acoustic_frequency_epoch_idx = 0

        stimulus_epoch_onsets: npt.NDArray[(t.Any,), np.float64] = trial_windows["epocs"][self.attenuation_epoch][
            "onset"
        ]
        stimulus_epoch_offsets: npt.NDArray[(t.Any,), np.float64] = trial_windows["epocs"][self.attenuation_epoch][
            "offset"
        ]
        acoustic_frequency_epoch_onsets: npt.NDArray[(t.Any,), np.float64] = trial_windows["epocs"][
            self.acoustic_frequency_epoch
        ]["onset"]
        attenuation_epoch_onsets: npt.NDArray[(t.Any,), np.float64] = trial_windows["epocs"][self.attenuation_epoch][
            "onset"
        ]

        total_stimulus_count = len(stimulus_epoch_onsets)

        session = structures.Session()

        for trial_idx, trial_number, trial_start_timestamp, trial_end_timestamp in (
            (idx + 1, int(trial_data[0]), trial_data[1], trial_data[2])
            for idx, trial_data in enumerate(
                zip(
                    trial_windows["epocs"][self.trial_epoch]["data"],
                    trial_windows["time_ranges"][TIME_RANGE_ONSET_IDX],
                    trial_windows["time_ranges"][TIME_RANGE_OFFSET_IDX],
                )
            )
        ):
            logger.debug(
                "Starting processing of trial number %s, start timestamp %s, end timestamp %s",
                trial_number,
                trial_start_timestamp,
                trial_end_timestamp,
            )
            trial = structures.Trial(
                trial_number=trial_number,
                start_timestamp=trial_start_timestamp,
                end_timestamp=trial_end_timestamp,
                excluded=False,
            )

            # Skip forward to the first stimulus inside the trial time range
            while stimulus_epoch_onsets[stimulus_epoch_idx] < trial_start_timestamp:
                stimulus_epoch_idx += 1

            logger.debug(
                "Found first stimulus onset after trial start: stimulus_epoch_idx: %s, timestamp: %s",
                stimulus_epoch_idx,
                stimulus_epoch_onsets[stimulus_epoch_idx],
            )

            while (
                stimulus_epoch_idx < total_stimulus_count
                and stimulus_epoch_onsets[stimulus_epoch_idx] < trial_end_timestamp
            ):
                logger.debug(
                    "Processing stimulus: stimulus_epoch_idx: %s, timestamp: %s",
                    stimulus_epoch_idx,
                    stimulus_epoch_onsets[stimulus_epoch_idx],
                )
                acoustic_frequency: t.Union[int, None] = None
                acoustic_attenuation: t.Union[int, None] = None

                # Skip forward to the next possible matching acoustic attenuation and frequency epocs for that stimulus
                while (
                    attenuation_epoch_onsets[attenuation_epoch_idx]
                    < stimulus_epoch_onsets[stimulus_epoch_idx] - EPOCH_TIMESTAMP_ERROR_MARGIN
                ):
                    attenuation_epoch_idx += 1

                while (
                    acoustic_frequency_epoch_onsets[acoustic_frequency_epoch_idx]
                    < stimulus_epoch_onsets[stimulus_epoch_idx] - EPOCH_TIMESTAMP_ERROR_MARGIN
                ):
                    acoustic_frequency_epoch_idx += 1

                # If within the 'margin of error' then record the acoustic frequency and attenuation values
                #  for the stimulus
                if (
                    abs(
                        acoustic_frequency_epoch_onsets[acoustic_frequency_epoch_idx]
                        - stimulus_epoch_onsets[stimulus_epoch_idx]
                    )
                    < EPOCH_TIMESTAMP_ERROR_MARGIN
                ):
                    acoustic_frequency = int(
                        trial_windows["epocs"][self.acoustic_frequency_epoch]["data"][acoustic_frequency_epoch_idx]
                    )

                if (
                    abs(attenuation_epoch_onsets[attenuation_epoch_idx] - stimulus_epoch_onsets[stimulus_epoch_idx])
                    < EPOCH_TIMESTAMP_ERROR_MARGIN
                ):
                    acoustic_attenuation = int(
                        trial_windows["epocs"][self.attenuation_epoch]["data"][attenuation_epoch_idx]
                    )

                tone = structures.Tone(
                    tone_start_timestamp=stimulus_epoch_onsets[stimulus_epoch_idx],
                    tone_end_timestamp=stimulus_epoch_offsets[stimulus_epoch_idx],
                    # @todo: consider using the start of the next stim as the actual offset
                    inter_tone_interval_end_timestamp=(
                        stimulus_epoch_offsets[stimulus_epoch_idx] + self.inter_tone_interval
                    ),
                    frequency=acoustic_frequency,
                    attenuation=acoustic_attenuation,
                )

                stimulus_epoch_idx += 1
                trial.tones_data.append(tone)

            session.trials_data.append(trial)

            # in_tone_data = np.zeros((tones_to_include * 2, 32), np.uint)
            # # out_tone_data = np.zeros((tones_to_include, 32), np.uint)
            # for tone_number, tone_offset in enumerate(
            #     filter(
            #         lambda x: (
            #             trial_windows["time_ranges"][TIME_RANGE_ONSET_IDX][trial_number]
            #             < x
            #             < trial_windows["time_ranges"][TIME_RANGE_OFFSET_IDX][trial_number]
            #         ),
            #         trial_windows["epocs"]["StiS"]["onset"],
            #     )
            # ):
            #     if tone_number >= tones_to_include:
            #         continue

            #     spike_cumulator_range = strictures.SpikeAccumulatorTimeRange(
            #         start=tone_offset + self.in_tone_capture_start_offset,
            #         end=tone_offset + self.in_tone_capture_end_offset,
            #     )
            #     while cspk_data["ts"][cspk_offset] < spike_cumulator_range.start:
            #         cspk_offset += 1

            #     while cspk_data["ts"][cspk_offset] < spike_cumulator_range.end:
            #         in_tone_data[tone_number * 2, cspk_data["chan"][cspk_offset][0] - 1] += 1
            #         cspk_offset += 1

            #     spike_cumulator_range = structures.SpikeAccumulatorTimeRange(
            #         start=tone_offset + self.tone_duration + self.out_tone_capture_start_offset,
            #         end=tone_offset + self.tone_duration + self.out_tone_capture_end_offset,
            #     )
            #     while cspk_data["ts"][cspk_offset] < spike_cumulator_range.start:
            #         cspk_offset += 1

            #     while cspk_data["ts"][cspk_offset] < spike_cumulator_range.end:
            #         in_tone_data[tone_number * 2 + 1, cspk_data["chan"][cspk_offset][0] - 1] += 1
            #         cspk_offset += 1

            # # print(in_tone_data)
            # # plt.rcParams["figure.figsize"] = [100 / 2.54, 80 / 2.54]
            # # plt.imshow(in_tone_data, cmap="hot", interpolation="nearest")
            # # plt.show()

        return session
