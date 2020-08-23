import typing as t
from dataclasses import dataclass

import numpy as np

import tdt

# Offset in the filtered["time ranges"] for start and end of the time ranges included
#  via a filtering search
# i.e. filtered = tdt.epoc_filter(data, "TriS", t=[-10, 20])
#  would have filtered["time_ranges"]
TIME_RANGE_ONSET_IDX = 0
TIME_RANGE_OFFSET_IDX = 1


# Time range
class SpikeAccumulatorTimeRange(t.NamedTuple):
    start: float
    end: float


@dataclass
class Tone:
    """
    Data for a single tone/inter-tone-interval pair inside a trial

    Attributes:
        tone_start_timestamp: Starting timestamp in the block for the tone
        tone_end_timestamp: Ending timestamp in the block for the tone
        inter_tone_interval_end_timestamp: Ending timestamp in the block for the end of the inter-tone interval
        frequency: Frequency (in Hz) of the tone
        attenuation: Attenuation (in dB) of the tone
    """

    tone_start_timestamp: float
    tone_end_timestamp: float
    inter_tone_interval_end_timestamps: float
    frequency: int
    amplitude: int


@dataclass
class Trial:
    """Basic structure for a trial in a session"""

    trial_number: int
    base_frequency: int
    alternate_frequency: int
    amplitudes: t.List[int]
    tones_data: t.List[Tone]
    excluded: bool


@dataclass
class IncludedTrial(Trial):
    """Holds data for a single trial of a session"""

    in_tone_spike_counts: t.List[np.nparray]
    out_tone_spike_counts: t.List[np.nparray]


@dataclass
class ExcludedTrial(Trial):
    """Holds data representing an excluded trial and why it was excluded"""

    exclusion_reason: str


@dataclass
class Session:
    """Holds data for a full session"""

    trial_count: int
    trials_data: t.List[t.Type[Trial]]
    # @todo: Add channel mapping, animal info, excluded trials, etc


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
    accoustic_frequency_epoch = "AFrq"

    # neural stim parameters
    reference_channels_bitmask_epoch = "ReBM"
    stimulation_channels_bitmask_epoch = "StBM"
    stimulation_current_epoch = "Curr"
    stimulation_frequency_epoch = "Freq"

    # Epochs of unknown meaning
    # block_start_epoch = "BloS"  # Not always present? Has no values?
    # unknown_1_epoch = "SweS"

    def __init__(self, *, block_path: str):
        self.block_path = block_path
        self.block_data = tdt.read_block(self.block_path, evtype=["epocs", "snips", "scalars"], nodata=1)

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
        trial_epoch: str = None,
        stimulus_epoch: str = None,
        attenuation_epoch: str = None,
        accoustic_frequency_epoch: str = None,
        reference_channels_bitmask_epoch: str = None,
        stimulation_channels_bitmask_epoch: str = None,
        stimulation_current_epoch: str = None,
        stimulation_frequency_epoch: str = None,
    ):
        self.trial_epoch = trial_epoch or self.trial_epoch
        self.stimulus_epoch = stimulus_epoch or self.stimulus_epoch
        self.attenuation_epoch = attenuation_epoch or self.attenuation_epoch
        self.accoustic_frequency_epoch = accoustic_frequency_epoch or self.accoustic_frequency_epoch
        self.reference_channels_bitmask_epoch = (
            reference_channels_bitmask_epoch or self.reference_channels_bitmask_epoch
        )
        self.stimulation_channels_bitmask_epoch = (
            stimulation_channels_bitmask_epoch or self.stimulation_channels_bitmask_epoch
        )
        self.stimulation_current_epoch = stimulation_current_epoch or self.stimulation_current_epoch

    def extract_spikes_from_trial(self, *, trial_start_offset: float, trial_end_offset: float) -> Session:
        """
        Extracts spike counts for each tone of each trial in the session

        Parameters
        ----------
        trial_start_offset :
            Offset from trial start from which to include spike data per tone
        trial_end_offset :
            Offset from trial start until which to include spike data per tone
            (note: any incomplete tones at the end are not included)

        Returns
        -------
        Session object for the testing session
            
        """
        tones_per_second: float = 1 / (self.tone_duration + self.inter_tone_interval)
        trial_windows = tdt.epoc_filter(self.block_data, "TriS", t=[START_OFFSET, END_OFFSET - START_OFFSET])

        # Skip the last tone as the end offset should always go a little past the end
        tone_count_to_include = int(((end_offset - start_offset) * tones_per_second) - 1)

        all_trials_data: t.List[np.nparray] = []
        cspk_offset = 0
        cspk_data = filtered["snips"]["CSPK"]
        cspk_length = len(cspk_data["ts"])
        np.set_printoptions(threshold=sys.maxsize)

        for trial_number in range(len(filtered["time_ranges"][0])):
            in_tone_data = np.zeros((tones_to_include * 2, 32), np.uint)
            # out_tone_data = np.zeros((tones_to_include, 32), np.uint)
            for tone_number, tone_offset in enumerate(
                filter(
                    lambda x: (filtered["time_ranges"][TIME_RANGE_ONSET_IDX][trial_number]
                    < x
                    < filtered["time_ranges"][TIME_RANGE_OFFSET_IDX][trial_number]),
                    filtered["epocs"]["StiS"]["onset"],
                )
            ):
                if tone_number >= tones_to_include:
                    continue

                spike_cumulator_range = SpikeCountRange(
                    start=tone_offset + IN_TONE_CAPTURE_START_OFFSET, end=tone_offset + IN_TONE_CAPTURE_END_OFFSET
                )
                while cspk_data["ts"][cspk_offset] < spike_cumulator_range.start:
                    cspk_offset += 1

                while cspk_data["ts"][cspk_offset] < spike_cumulator_range.end:
                    in_tone_data[tone_number * 2, cspk_data["chan"][cspk_offset][0] - 1] += 1
                    cspk_offset += 1

                spike_cumulator_range = SpikeCountRange(
                    start=tone_offset + TONE_DURATION + OUT_TONE_CAPTURE_START_OFFSET,
                    end=tone_offset + TONE_DURATION + OUT_TONE_CAPTURE_END_OFFSET,
                )
                while cspk_data["ts"][cspk_offset] < spike_cumulator_range.start:
                    cspk_offset += 1

                while cspk_data["ts"][cspk_offset] < spike_cumulator_range.end:
                    in_tone_data[tone_number * 2 + 1, cspk_data["chan"][cspk_offset][0] - 1] += 1
                    cspk_offset += 1

            print(in_tone_data)
            plt.rcParams["figure.figsize"] = [100 / 2.54, 80 / 2.54]
            plt.imshow(in_tone_data, cmap="hot", interpolation="nearest")
            plt.show()
