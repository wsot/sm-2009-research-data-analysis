import collections
import logging
import pathlib
import typing as t

import nptyping as npt
import numpy as np

from . import _base as base

logger = logging.getLogger(__name__)


class ChannelRemapper(base.BaseDataProcessor):
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
