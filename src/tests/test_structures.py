# type: ignore
# noqa: S101
import pathlib
import typing as t

import numpy as np
import pytest

from lib import structures

SentinelObject = object()


class TestChannelRemapper:
    def test_rejects_duplicate_source_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 2), (2, 3))

        with pytest.raises(ValueError, match="duplicate source channels"):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_duplicate_destination_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 2), (3, 2))

        with pytest.raises(ValueError, match="duplicate destination channels"):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_missing_source_channels(self) -> None:
        invalid_mapping = ((1, 1,), (3, 2), (4, 3))

        with pytest.raises(ValueError, match="source channel numbers exceed max channel number"):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_missing_destination_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 3), (3, 4))

        with pytest.raises(ValueError, match="destination channel numbers exceed max channel number"):
            structures.ChannelRemapper(invalid_mapping)

    def test_remaps_correctly_with_valid_map(self) -> None:
        valid_mapping = ((1, 2,), (2, 1), (3, 3))
        in_array = np.array(((100, 300, 500), (200, 400, 600)))

        remapper = structures.ChannelRemapper(valid_mapping)
        out_array = remapper.transform(in_array)
        expected_array = np.array(((300, 100, 500), (400, 200, 600)))

        assert (out_array == expected_array).all()

    @pytest.mark.parametrize("header_columns", ["TDT\tMapping", "TDT\tMapped"])
    def test_can_read_from_file(self, fs, monkeypatch, header_columns) -> None:
        call_count = [0]
        expected_mapping = ((1, 2), (3, 1), (2, 3))
        fs.create_file(
            "/my/data/channel_map.txt",
            contents=(
                f"""{header_columns}
1\t2
3\t1
2\t3
"""
            ),
        )

        def mock_init(self: structures.ChannelRemapper, channel_mapping: t.Any) -> None:
            assert tuple(channel_mapping) == expected_mapping
            call_count[0] += 1

        monkeypatch.setattr(structures.ChannelRemapper, "__init__", mock_init)
        channel_remapper = structures.ChannelRemapper.from_filename("/my/data/channel_map.txt")
        assert isinstance(channel_remapper, structures.ChannelRemapper)
        assert call_count[0] == 1

    def test_rejects_invalid_file_headers(self, fs, monkeypatch) -> None:
        call_count = [0]
        expected_mapping = ((1, 2), (3, 1), (2, 3))
        fs.create_file(
            "/my/data/channel_map.txt",
            contents=(
                """Not\tValid
1\t2
3\t1
2\t3
"""
            ),
        )

        def mock_init(self: structures.ChannelRemapper, channel_mapping: t.Any) -> None:
            assert tuple(channel_mapping) == expected_mapping
            call_count[0] += 1

        monkeypatch.setattr(structures.ChannelRemapper, "__init__", mock_init)

        with pytest.raises(ValueError, match="Invalid header line"):
            structures.ChannelRemapper.from_filename("/my/data/channel_map.txt")

        assert call_count[0] == 0

    @pytest.mark.parametrize("filename", ["channel map", "Channel map", "Channel Mapping"])
    def test_autofind_finds_valid_filenames(self, fs, monkeypatch, filename) -> None:
        call_count = [0]
        fs.create_file(f"/my/data/{filename}.txt")

        def mock_from_path(path: pathlib.Path) -> t.Any:
            call_count[0] += 1
            return SentinelObject

        monkeypatch.setattr(structures.ChannelRemapper, "from_path", mock_from_path)
        return_val = structures.ChannelRemapper.from_autofind_in_path(pathlib.Path("/my/data/"))
        assert return_val == SentinelObject

        assert call_count[0] == 1

    def test_autofind_returns_none_if_no_file_found(self, fs, monkeypatch) -> None:
        call_count = [0]
        fs.create_file("/my/data/not_valid_map_name.txt")

        def mock_from_path(path: pathlib.Path) -> None:
            call_count[0] += 0

        monkeypatch.setattr(structures.ChannelRemapper, "from_path", mock_from_path)
        channel_remapper = structures.ChannelRemapper.from_autofind_in_path(pathlib.Path("/my/data/"))
        assert channel_remapper is None
        assert call_count[0] == 0
