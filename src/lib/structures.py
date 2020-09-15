"""
Data structures for representing the processed data from testing/trials
"""
import logging
import typing as t
from dataclasses import dataclass

import nptyping as npt
import numpy as np

logger = logging.getLogger(__name__)


AcousticAttenuation = t.NewType("AcousticAttenuation", int)
AcousticFrequency = t.NewType("AcousticFrequency", int)


class TdtTimestamp(np.float64):
    """Wrapper type for TdtTimestamp to make sure they are used correctly"""

    # @todo: How can `np.float64 == TdtTimestamp(...)` be made to fail type checking
    #
    # @todo: Make these actually work properly, because right now they don't even type check properly
    # and it is frustrating
    #
    # @todo: Check how much performance impact there is using these classes, cos it ain't zero
    def __add__(self, other: "TdtRelativeTimestamp") -> "TdtTimestamp":
        if isinstance(other, TdtRelativeTimestamp):
            return TdtTimestamp(np.float64(self) + np.float64(other))
        raise TypeError("Only TdtRelativeTimestamps can be added to TdtTimestamps")

    def __sub__(
        self, other: t.Union["TdtTimestamp", "TdtRelativeTimestamp"]
    ) -> t.Union["TdtTimestamp", "TdtRelativeTimestamp"]:
        if isinstance(other, TdtRelativeTimestamp):
            return TdtTimestamp(np.float64(self) - np.float64(other))
        elif isinstance(other, TdtTimestamp):
            return TdtRelativeTimestamp(np.float64(self) - np.float64(other))
        raise TypeError("Only TdtRelativeTimestamps or TdtTimestamps be subtracted from TdtTimestamps")


class TdtRelativeTimestamp(np.float64):
    """Wrapper type for TdtRelativeTimestamp to make sure they are used correctly"""

    def __add__(
        self, other: t.Union["TdtTimestamp", "TdtRelativeTimestamp"]
    ) -> t.Union["TdtTimestamp", "TdtRelativeTimestamp"]:
        if isinstance(other, TdtTimestamp):
            return other + self
        elif isinstance(other, TdtRelativeTimestamp):
            return TdtRelativeTimestamp(np.float64(self) + np.float64(other))

        raise TypeError("Only TdtRelativeTimestamps or TdtTimestamps be added to TdtRelativeTimestamps")

    def __sub__(self, other: "TdtRelativeTimestamp") -> "TdtRelativeTimestamp":
        if isinstance(other, TdtRelativeTimestamp):
            return TdtRelativeTimestamp(np.float64(self) + np.float64(other))
        raise TypeError("Only TdtRelativeTimestamps can be subtracted from to TdtRelativeTimestamps")

    def __mul__(self, other: t.Any) -> "TdtRelativeTimestamp":
        return TdtRelativeTimestamp(np.float64(self) * other)

    def __div__(self, other: t.Any) -> "TdtRelativeTimestamp":
        return TdtRelativeTimestamp(np.float64(self) / other)


# Time range
class SpikeAccumulatorTimeRange(t.NamedTuple):
    start: np.float64
    end: np.float64


@dataclass
class Stimulus:
    """
    Base data structure for a stimulus inside a trial

    Attributes:
        stimulus_start_timestamp: Starting timestamp in the block for the stimulus
        stimulus_end_timestamp: Ending timestamp in the block for the stimulus
        inter_stimulus_interval_end_timestamp: Ending timestamp in the block for the end of the inter-stimulus interval

        stimulus_start_timestamp: Starting timestamp in the block for the stimulus
        stimulus_end_timestamp: Ending timestamp in the block for the stimulus
        inter_stimulus_interval_end_timestamp: Ending timestamp in the block for the end of the inter-stimulus interval
    """

    stimulus_start_timestamp: TdtTimestamp
    stimulus_end_timestamp: TdtTimestamp
    inter_stimulus_interval_end_timestamp: TdtTimestamp

    stimulus_start_relative_timestamp: TdtRelativeTimestamp
    stimulus_end_relative_timestamp: TdtRelativeTimestamp
    inter_stimulus_interval_end_relative_timestamp: TdtRelativeTimestamp


@dataclass
class Tone(Stimulus):
    """
    Data for a single tone/inter-tone-interval pair inside a trial

    Attributes:
        frequency: Frequency (in Hz) of the tone
        attenuation: Attenuation (in dB) of the tone
    """

    frequency: t.Union[AcousticFrequency, None]
    attenuation: t.Union[AcousticAttenuation, None]


class Trial:
    """Basic structure for a trial in a session"""

    trial_number: int
    start_timestamp: TdtTimestamp
    end_timestamp: TdtTimestamp
    excluded: bool

    stimuli: t.List[Stimulus]


class AcousticTrial(Trial):
    """Acoustic Trial structure"""

    base_frequency: t.Union[AcousticFrequency, None]
    alternate_frequency: t.Union[AcousticFrequency, None]
    amplitudes: t.List[AcousticAttenuation]

    def __init__(
        self, *, trial_number: int, start_timestamp: np.float64, end_timestamp: np.float64, excluded: bool
    ) -> None:
        self.base_frequency = None
        self.alternate_frequency = None

        self.trial_number = trial_number
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.excluded = excluded
        self.amplitudes = []
        self.stimuli = []


class IncludedTrial(Trial):
    """Holds data for a single trial of a session"""

    # These are stored as [stimulus number][channel number]
    # So to get the data for all channels on the first stimulus:
    #  in_stimulus_spike_counts[0]
    # To get the data for channel 10 on all stimuli:
    #  in_stimulus_spike_counts[:, 9]
    in_stimulus_spike_counts: npt.NDArray[(t.Any, t.Any), np.int32] = []
    out_stimulus_spike_counts: npt.NDArray[(t.Any, t.Any), np.int32] = []

    # # Raw spike timestamps in structure:
    # #  in_tone_spike_timestamps[stimulus number][channel number][array of timestamps]
    # in_tone_spike_timestamps: t.List[t.List[npt.NDArray[(t.Any, ), np.int32]]] = []
    # out_tone_spike_timestamps: t.List[t.List[npt.NDArray[(t.Any, ), np.int32]]] = []


class ExcludedTrial(Trial):
    """Holds data representing an excluded trial and why it was excluded"""

    exclusion_reason: str = ""


class IncludedAcousticTrial(IncludedTrial, AcousticTrial):
    pass


class ExcludedAcousticTrial(ExcludedTrial, AcousticTrial):
    pass


class Session:
    """Holds data for a full session"""

    @property
    def trial_count(self) -> int:
        """Get the number of trials in the session

        Returns
        -------
        int
            Number of trials in the session
        """
        return len(self.trials)

    trials: t.List[Trial]
    # @todo: Add channel mapping, animal info, excluded trials, etc

    def __init__(self) -> None:
        self.trials = []
