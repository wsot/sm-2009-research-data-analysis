"""
Data structures for representing the processed data from testing/trials
"""
import logging
import typing as t
from dataclasses import dataclass

import nptyping as npt
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Time range
class SpikeAccumulatorTimeRange(t.NamedTuple):
    start: np.float64
    end: np.float64


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

    tone_start_timestamp: np.float64
    tone_end_timestamp: np.float64
    inter_tone_interval_end_timestamp: np.float64
    frequency: t.Union[int, None]
    attenuation: t.Union[int, None]


class Trial:
    """Basic structure for a trial in a session"""

    trial_number: int
    start_timestamp: np.float64
    end_timestamp: np.float64
    excluded: bool
    base_frequency: int
    alternate_frequency: int
    amplitudes: t.List[int]
    tones_data: t.List[Tone]

    def __init__(
        self, *, trial_number: int, start_timestamp: np.float64, end_timestamp: np.float64, excluded: bool
    ) -> None:
        self.trial_number = trial_number
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.excluded = excluded
        self.amplitudes = []
        self.tones_data = []


class IncludedTrial(Trial):
    """Holds data for a single trial of a session"""

    in_tone_spike_counts: t.List[npt.NDArray[(t.Any,), np.int32]] = []
    out_tone_spike_counts: t.List[npt.NDArray[(t.Any,), np.int32]] = []


class ExcludedTrial(Trial):
    """Holds data representing an excluded trial and why it was excluded"""

    exclusion_reason: str = ""


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
        return len(self.trials_data)

    trials_data: t.List[Trial]
    # @todo: Add channel mapping, animal info, excluded trials, etc

    def __init__(self) -> None:
        self.trials_data = []
