"""
Data structures for representing the processed data from testing/trials
"""
import logging
import typing as t
from dataclasses import dataclass

import nptyping as npt
import numpy as np

logger = logging.getLogger(__name__)


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

    stimulus_start_timestamp: np.float64
    stimulus_end_timestamp: np.float64
    inter_stimulus_interval_end_timestamp: np.float64

    stimulus_start_relative_timestamp: np.float64
    stimulus_end_relative_timestamp: np.float64
    inter_stimulus_interval_end_relative_timestamp: np.float64


@dataclass
class Tone(Stimulus):
    """
    Data for a single tone/inter-tone-interval pair inside a trial

    Attributes:
        frequency: Frequency (in Hz) of the tone
        attenuation: Attenuation (in dB) of the tone
    """

    frequency: t.Union[int, None]
    attenuation: t.Union[int, None]


class Trial:
    """Basic structure for a trial in a session"""

    trial_number: int
    start_timestamp: np.float64
    end_timestamp: np.float64
    excluded: bool

    stimuli: t.List[Stimulus]


class AcousticTrial(Trial):
    """Acoustic Trial structure"""

    base_frequency: t.Union[int, None]
    alternate_frequency: t.Union[int, None]
    amplitudes: t.List[int]

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

    in_stimulus_spike_counts: t.List[npt.NDArray[(t.Any,), np.int32]] = []
    out_stimulus_spike_counts: t.List[npt.NDArray[(t.Any,), np.int32]] = []

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
