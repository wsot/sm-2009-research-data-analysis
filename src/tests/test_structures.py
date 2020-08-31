import typing as t

import nptyping as npt
import numpy as np
import pytest

from lib import structures


class TestChannelRemapper:
    def test_rejects_duplicate_source_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 2), (2, 3))

        with pytest.raises(ValueError):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_duplicate_destination_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 2), (3, 2))

        with pytest.raises(ValueError):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_missing_source_channels(self) -> None:
        invalid_mapping = ((1, 1,), (3, 2), (4, 3))

        with pytest.raises(ValueError):
            structures.ChannelRemapper(invalid_mapping)

    def test_rejects_missing_destination_channels(self) -> None:
        invalid_mapping = ((1, 1,), (2, 3), (3, 4))

        with pytest.raises(ValueError):
            structures.ChannelRemapper(invalid_mapping)

    def test_remaps_correctly_with_valid_map(self) -> None:
        valid_mapping = ((1, 2,), (2, 1), (3, 3))
        in_array = np.array(((100, 300, 500), (200, 400, 600)))

        remapper = structures.ChannelRemapper(valid_mapping)
        out_array = remapper.transform(in_array)
        expected_array = np.array(((300, 100, 500), (400, 200, 600)))

        assert (out_array == expected_array).all()

    def test_can_read_from_file(self, fs, monkeypatch) -> None:
        call_count = [0]
        expected_mapping = ((1, 2), (3, 1), (2, 3))
        fs.create_file(
            "/my/data/channel_map.txt",
            contents=(
                """TDT\tChannel
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
        structures.ChannelRemapper.from_filename("/my/data/channel_map.txt")

        assert call_count[0] == 1
