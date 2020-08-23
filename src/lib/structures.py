import typing as t

import numpy as np


# Time range
class SpikeAccumulatorTimeRange(t.NamedTuple):
    start: float
    end: float


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


class Trial:
    """Basic structure for a trial in a session"""

    trial_number: int
    excluded: bool
    base_frequency: int
    alternate_frequency: int
    amplitudes: t.List[int] = []
    tones_data: t.List[Tone] = []


class IncludedTrial(Trial):
    """Holds data for a single trial of a session"""

    in_tone_spike_counts: t.List[np.nparray] = []
    out_tone_spike_counts: t.List[np.nparray] = []


class ExcludedTrial(Trial):
    """Holds data representing an excluded trial and why it was excluded"""

    exclusion_reason: str


class Session:
    """Holds data for a full session"""

    @property
    def trial_count(self) -> int:
        return len(self.trials_data)

    trials_data: t.List[t.Type[Trial]] = []
    # @todo: Add channel mapping, animal info, excluded trials, etc
