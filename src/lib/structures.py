"""
Data structures for representing the processed data from testing/trials
"""
import abc
import collections
import enum
import logging
import pathlib
import re
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


class DataProcessor(abc.ABC):
    """Base class for an object that can take the output array from a trial and modify it in some
    way, returning a new array of equal size and meaning"""

    @abc.abstractmethod
    def transform(self, array: npt.NDArray[(t.Any, t.Any), t.Any]) -> npt.NDArray[(t.Any, t.Any), t.Any]:
        pass


class ChannelRemapper(DataProcessor):
    """
    Remaps 'channels' in an array of data[:, channel_number]. Requires that channels be numbered
    1...n
    """

    channel_map: t.Collection[t.Tuple[int, int]]

    def __init__(self, channel_map: t.Collection[t.Tuple[int, int]]) -> None:
        self._set_channel_map(channel_map)

    def _set_channel_map(self, channel_map: t.Collection[t.Tuple[int, int]]) -> None:
        """Validate provided channel map and store if valid

        Parameters
        ----------
        channel_map : t.Collection[t.Collection[int, int]]
            Collection of mappings ((from channel, to channel), (from channel, to channel), ...)

        Raises
        ------
        ValueError
            Raised if channel map includes duplicate source or destination channels
        """
        max_channel = len(channel_map)
        # To track the frequency of each channel number appearing for error handling
        src_channels_freq: t.DefaultDict[int, int] = collections.defaultdict(int)
        dst_channels_freq: t.DefaultDict[int, int] = collections.defaultdict(int)
        src_channels_over_max = set()
        dst_channels_over_max = set()

        for src_channel, dst_channel in channel_map:
            logging.debug("Mapping source channel %s to destination %s", src_channel, dst_channel)
            if src_channel > max_channel:
                src_channels_over_max.add(str(src_channel))
            if dst_channel > max_channel:
                dst_channels_over_max.add(str(dst_channel))
            src_channels_freq[src_channel] += 1
            dst_channels_freq[dst_channel] += 1

        error_messages: t.List[str] = []
        duplicated_src_message = ", ".join([str(channel) for channel, freq in src_channels_freq.items() if freq > 1])
        if duplicated_src_message:
            error_messages.append(f"Provided channel map includes duplicate source channels: {duplicated_src_message}")

        duplicated_dst_message = ", ".join([str(channel) for channel, freq in dst_channels_freq.items() if freq > 1])
        if duplicated_dst_message:
            error_messages.append(
                f"Provided channel map includes duplicate destination channels: {duplicated_dst_message}"
            )

        if src_channels_over_max:
            error_messages.append(
                "One or more source channel numbers exceed max channel number "
                f"({max_channel}): {', '.join(src_channels_over_max)}"
            )
        if dst_channels_over_max:
            error_messages.append(
                "One or more destination channel numbers exceed max channel number "
                f"({max_channel}): {', '.join(dst_channels_over_max)}"
            )

        if error_messages:
            raise ValueError(f"Channel map contains errors: {'; '.join(error_messages)}")

        self.channel_map = channel_map

    def transform(self, in_array: npt.NDArray[(t.Any, t.Any), t.Any]) -> npt.NDArray[(t.Any, t.Any), t.Any]:
        # Check the mapping is exhaustive for all channels
        if len(self.channel_map) != in_array.shape[1]:
            raise ValueError("Number of channels in the array does not match number of channels in the map")

        out_array = np.empty(in_array.shape)

        # List of channels
        for src_channel, dst_channel in self.channel_map:
            out_array[:, dst_channel - 1] = in_array[:, src_channel - 1]

        return out_array

    @classmethod
    def from_filename(cls, filename: str) -> "ChannelRemapper":
        return cls.from_path(pathlib.Path(filename))

    @classmethod
    def from_path(cls, path: pathlib.Path) -> "ChannelRemapper":
        with path.open() as f:
            return cls.from_file(f)

    @classmethod
    def from_file(cls, f: t.TextIO) -> "ChannelRemapper":
        channel_map = []
        try:
            header_line = tuple(f.readline().strip().split("\t"))
            if header_line not in (("TDT", "Mapping"), ("TDT", "Mapped")) != header_line:
                raise ValueError(f"Invalid header line: {header_line}")
            for line in f:
                src_channel, dst_channel = line.strip().split("\t")
                channel_map.append((int(src_channel), int(dst_channel)))
        except ValueError as e:
            raise ValueError(f"Data structure of provided channel map is not valid: {e}")

        return cls(channel_map)

    @classmethod
    def from_autofind_in_path(
        cls, path: pathlib.Path, mapping_filename_prefix: str = "channel map"
    ) -> t.Optional["ChannelRemapper"]:
        f: t.Optional[pathlib.Path] = None
        for f in path.iterdir():
            if f.is_file() and f.name.lower().startswith(mapping_filename_prefix.lower()):
                break

        if f is not None:
            logger.info("Loading channel mappings from %s", f.name)
            return cls.from_path(f)

        return None


class ExclusionDataType(enum.Enum):
    NEURAL_DATA = 1
    HR_DATA = 2


class ExclusionTrialsType(enum.Enum):
    ACOUSTIC_TRIALS = 1
    ELECTRICAL_TRIALS = 2


@dataclass
class TrialExclusion:
    data_types: t.List[ExclusionDataType]
    trials_types: t.List[ExclusionTrialsType]
    start_offset: TdtTimestamp
    end_offset: t.Optional[TdtTimestamp]
    reason: str

    @classmethod
    def from_autofind_in_path(
        cls, path: pathlib.Path, exclusion_file_prefix: str = "exclude "
    ) -> t.Iterable["TrialExclusion"]:

        exclusions: t.List["TrialExclusion"] = []
        exclusion_files_by_data_type: t.DefaultDict[ExclusionDataType, t.List[str]] = collections.defaultdict(list)
        for p in path.parents:
            for f in p.iterdir():
                if f.is_file() and f.name.lower().startswith(exclusion_file_prefix.lower()):
                    exclusion = cls.from_path(f)
                    if exclusion is not None:
                        exclusions.append(exclusion)
                        for data_type in exclusion.data_types:
                            exclusion_files_by_data_type[data_type].append(f.absolute().as_posix())

        for k, v in exclusion_files_by_data_type.items():
            if len(v) > 1:
                logger.warning(f"More than one exclusion file for data type {k.name}: {', '.join(v)}")

        return exclusions

    @classmethod
    def from_file(
        cls, f: t.TextIO, data_types: t.List[ExclusionDataType], trials_types: t.List[ExclusionTrialsType]
    ) -> "TrialExclusion":
        start_offset = TdtTimestamp(0.0)
        end_offset: t.Optional[TdtTimestamp] = None
        reason: t.List[str] = []

        first_line = f.readline()
        matches = re.match(r"Exclude after:\s*(\d+)s", f.readline())
        if matches:
            start_offset = TdtTimestamp(np.float64(matches.group(1)))
        else:
            reason.append(first_line)

        reason.extend(f)
        reason_text = "\n".join(reason).strip()

        return TrialExclusion(
            data_types=data_types,
            trials_types=trials_types,
            start_offset=start_offset,
            end_offset=end_offset,
            reason=reason_text,
        )

    @classmethod
    def from_path(cls, path: pathlib.Path) -> t.Optional["TrialExclusion"]:
        data_types: t.List[ExclusionDataType] = []
        trials_types: t.List[ExclusionTrialsType] = []
        if "neural" in path.name.lower():
            data_types.append(ExclusionDataType.NEURAL_DATA)
        elif "aggregration" in path.name.lower():
            return None
        else:
            logger.warning(f"Unknown exclusion type: {path.absolute}")

        with path.open() as f:
            return cls.from_file(f, data_types, trials_types)
