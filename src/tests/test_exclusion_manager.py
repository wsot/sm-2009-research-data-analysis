# type: ignore
import io
import logging
import pathlib

import pytest

from lib import exclusion_manager, structures


class TestExclusionManager:
    def test_creates_from_empty_file_with_no_reason_and_no_end_offset(self):
        data = ""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        trial_exclusion = exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)
        assert isinstance(trial_exclusion, exclusion_manager.TrialExclusion)
        assert trial_exclusion.data_types == data_types
        assert trial_exclusion.trials_types == trials_types
        assert trial_exclusion.reason == ""
        assert trial_exclusion.start_offset == structures.TdtTimestamp(0.0)
        assert trial_exclusion.end_offset is None

    def test_reads_start_and_end_offset_from_file(self):
        data = """Exclude before: 9500s
Exclude after: 5700s"""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        trial_exclusion = exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)
        assert isinstance(trial_exclusion, exclusion_manager.TrialExclusion)
        assert trial_exclusion.data_types == data_types
        assert trial_exclusion.trials_types == trials_types
        assert trial_exclusion.reason == ""
        assert trial_exclusion.start_offset == structures.TdtTimestamp(5700.0)
        assert trial_exclusion.end_offset == structures.TdtTimestamp(9500.0)

    def test_reads_non_offset_lines_as_reason_in_any_position(self):
        data = """Reason part 1
Exclude before: 9500s
Reason part 2
Exclude after: 5700s
Reason part 3
"""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        trial_exclusion = exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)
        assert isinstance(trial_exclusion, exclusion_manager.TrialExclusion)
        assert trial_exclusion.data_types == data_types
        assert trial_exclusion.trials_types == trials_types
        assert trial_exclusion.reason == "Reason part 1\nReason part 2\nReason part 3"
        assert trial_exclusion.start_offset == structures.TdtTimestamp(5700.0)
        assert trial_exclusion.end_offset == structures.TdtTimestamp(9500.0)

    def test_reads_reason_if_no_start_or_end(self):
        data = """Reason part 1
Reason part 2
Reason part 3
"""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        trial_exclusion = exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)
        assert isinstance(trial_exclusion, exclusion_manager.TrialExclusion)
        assert trial_exclusion.data_types == data_types
        assert trial_exclusion.trials_types == trials_types
        assert trial_exclusion.reason == "Reason part 1\nReason part 2\nReason part 3"
        assert trial_exclusion.start_offset == structures.TdtTimestamp(0.0)
        assert trial_exclusion.end_offset is None

    def test_raises_exception_on_duplicate_exclude_before(self):
        data = """Exclude before: 9500s
Exclude before: 9500s
Reason part 3
"""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        with pytest.raises(exclusion_manager.MultipleExcludeBefore):
            exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)

    def test_raises_exception_on_duplicate_exclude_after(self):
        data = """Exclude after: 9500s
Exclude after: 9500s
Reason part 3
"""
        data_types = [exclusion_manager.ExclusionDataType.NEURAL_DATA]
        trials_types = [exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS]

        with pytest.raises(exclusion_manager.MultipleExcludeAfter):
            exclusion_manager.TrialExclusion.from_file(io.StringIO(data), data_types, trials_types)

    def test_autofind_searches_file_hierarchy(self, fs) -> None:
        fs.create_file("/my/data/more/nested/exclude.txt", contents="Exclude after: 9500\nReason part 3")
        fs.create_file("/my/exclude from neural data - all.txt", contents="Reason part 3")
        fs.create_file("/exclude from results aggregation - all.txt")

        exclusions = exclusion_manager.TrialExclusion.from_autofind_in_path(pathlib.Path("/my/data/more/nested/"))
        assert len(exclusions) == 3

    def test_autofind_warns_if_multiple_exclusion_files_of_same_type(self, fs, caplog) -> None:
        fs.create_file("/my/data/more/nested/exclude.txt")
        fs.create_file("/my/exclude from neural data - all.txt")
        fs.create_file("/exclude from neural data - all.txt")

        with caplog.at_level(logging.WARNING):
            exclusions = exclusion_manager.TrialExclusion.from_autofind_in_path(pathlib.Path("/my/data/more/nested"))
            assert any(
                "More than one exclusion file for data type NEURAL_DATA" in record.message for record in caplog.records
            )
        assert len(exclusions) == 3

    def test_autofind_returns_empty_list_if_no_exclusions_found(self, fs, monkeypatch) -> None:
        call_count = [0]
        fs.create_file("/my/data/more/nested/not_exclusions.txt")
        fs.create_file("/my/data/not_exclusions.txt")
        fs.create_file("/my/not_exclusions.txt")

        def mock_from_path(path: pathlib.Path) -> None:
            call_count[0] += 0

        monkeypatch.setattr(exclusion_manager.TrialExclusion, "from_path", mock_from_path)
        exclusions = exclusion_manager.TrialExclusion.from_autofind_in_path(pathlib.Path("/my/data/more/nested"))
        assert exclusions == []
        assert call_count[0] == 0

    @pytest.mark.parametrize(
        "filename",
        [
            "exclude from neural data.txt",
            "exclude from neural data - all.txt",
            "exclude from neural data - partial.txt",
        ],
    )
    def test_from_path_matches_neural_data(self, fs, filename) -> None:
        fs.create_file(filename)
        exclusion = exclusion_manager.TrialExclusion.from_path(pathlib.Path(filename))
        assert exclusion.data_types == {exclusion_manager.ExclusionDataType.NEURAL_DATA}
        assert exclusion.trials_types == {
            exclusion_manager.ExclusionTrialsType.ELECTRICAL_TRIALS,
            exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS,
        }

    @pytest.mark.parametrize(
        "filename",
        [
            "exclude.txt",
            "exclude from results aggregation.txt",
            "exclude from results aggregation - all.txt",
            "exclude from results aggregation - all with message.txt",
        ],
    )
    def test_from_path_matches_full(self, fs, filename) -> None:
        fs.create_file(filename)
        exclusion = exclusion_manager.TrialExclusion.from_path(pathlib.Path(filename))
        assert exclusion.data_types == {
            exclusion_manager.ExclusionDataType.NEURAL_DATA,
            exclusion_manager.ExclusionDataType.HR_DATA,
        }
        assert exclusion.trials_types == {
            exclusion_manager.ExclusionTrialsType.ELECTRICAL_TRIALS,
            exclusion_manager.ExclusionTrialsType.ACOUSTIC_TRIALS,
        }

    @pytest.mark.parametrize(
        "filename",
        [
            "exclude from results aggregation - electrical.txt",
            "exclude from results aggregation - electrical with message.txt",
            "exclude from results aggregation - partial electrical with message.txt",
        ],
    )
    def test_from_path_matches_electrical(self, fs, filename) -> None:
        fs.create_file(filename)
        exclusion = exclusion_manager.TrialExclusion.from_path(pathlib.Path(filename))
        assert exclusion.data_types == {
            exclusion_manager.ExclusionDataType.NEURAL_DATA,
            exclusion_manager.ExclusionDataType.HR_DATA,
        }
        assert exclusion.trials_types == {
            exclusion_manager.ExclusionTrialsType.ELECTRICAL_TRIALS,
        }

    @pytest.mark.parametrize(
        "filename",
        [
            "exclude from cf saving.txt",
            "exclude bulk reprocessing.txt",
            "exclude from map generation.txt",
            "exclude some unknown type.txt",
        ],
    )
    def test_from_path_skips_unknown(self, fs, filename) -> None:
        fs.create_file(filename)
        exclusion = exclusion_manager.TrialExclusion.from_path(pathlib.Path(filename))
        assert exclusion is None
