import collections
import enum
import itertools
import logging
import pathlib
import re
import typing as t
from dataclasses import dataclass

import numpy as np

from . import structures

logger = logging.getLogger(__name__)


class InvalidExclusionFile(Exception):
    """Base class for exceptions when trying to parse an exclusion file"""

    pass


class MultipleExcludeAfter(InvalidExclusionFile):
    """Raised when multiple lines match `Exclude after:` format"""

    pass


class MultipleExcludeBefore(InvalidExclusionFile):
    """Raised when multiple lines match `Exclude before:` format"""

    pass


class ExclusionDataType(enum.Enum):
    NEURAL_DATA = 1
    HR_DATA = 2


ALL_EXCLUSION_DATA_TYPES = tuple(dt for dt in ExclusionDataType)


class ExclusionTrialsType(enum.Enum):
    ACOUSTIC_TRIALS = 1
    ELECTRICAL_TRIALS = 2


ALL_EXCLUSION_TRIALS_TYPES = tuple(tt for tt in ExclusionTrialsType)


@dataclass
class TrialExclusion:
    data_types: t.Set[ExclusionDataType]
    trials_types: t.Set[ExclusionTrialsType]
    start_offset: structures.TdtTimestamp
    end_offset: t.Optional[structures.TdtTimestamp]
    reason: str

    @classmethod
    def from_autofind_in_path(
        cls, path: pathlib.Path, exclusion_file_prefix: str = "exclude"
    ) -> t.Iterable["TrialExclusion"]:

        exclusions: t.List["TrialExclusion"] = []
        exclusion_files_by_data_type: t.DefaultDict[ExclusionDataType, t.List[str]] = collections.defaultdict(list)
        for p in itertools.chain((path,), path.parents):
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
        cls, f: t.TextIO, data_types: t.Set[ExclusionDataType], trials_types: t.Set[ExclusionTrialsType]
    ) -> "TrialExclusion":
        start_offset = None
        end_offset: t.Optional[structures.TdtTimestamp] = None
        reason: t.List[str] = []

        for line in f:
            matches = re.match(r"Exclude after:\s+(\d+)s?", line, flags=re.IGNORECASE)
            if matches:
                if start_offset is not None:
                    raise MultipleExcludeAfter
                start_offset = structures.TdtTimestamp(np.float64(matches.group(1)))
                continue
            matches = re.match(r"Exclude before:\s+(\d+)s", line, flags=re.IGNORECASE)
            if matches:
                if end_offset is not None:
                    raise MultipleExcludeBefore
                end_offset = structures.TdtTimestamp(np.float64(matches.group(1)))
                continue
            reason.append(line)

        start_offset = start_offset if start_offset is not None else structures.TdtTimestamp(0.0)
        reason_text = "".join(reason).strip()

        return TrialExclusion(
            data_types=data_types,
            trials_types=trials_types,
            start_offset=start_offset,
            end_offset=end_offset,
            reason=reason_text,
        )

    @classmethod
    def from_path(cls, path: pathlib.Path, exclusion_file_prefix: str = "exclude") -> t.Optional["TrialExclusion"]:
        data_types: t.Set[ExclusionDataType]
        trials_types: t.Set[ExclusionTrialsType]

        filename_remainder = path.name.lower()[
            len(exclusion_file_prefix) :  # noqa: E203  # black is making a mess here?
        ]

        if filename_remainder.endswith(".txt"):
            filename_remainder = filename_remainder[: -len(".txt")]

        if filename_remainder == "":
            data_types = set(ALL_EXCLUSION_DATA_TYPES)
            trials_types = set(ALL_EXCLUSION_TRIALS_TYPES)

        elif "neural data" in filename_remainder:
            data_types = {ExclusionDataType.NEURAL_DATA}
            trials_types = set(ALL_EXCLUSION_TRIALS_TYPES)

        elif (
            "results aggregation - electrical" in filename_remainder
            or "results aggregation - partial electrical" in filename_remainder
        ):
            data_types = set(ALL_EXCLUSION_DATA_TYPES)
            trials_types = {ExclusionTrialsType.ELECTRICAL_TRIALS}

        elif "results aggregation" in filename_remainder:
            data_types = set(ALL_EXCLUSION_DATA_TYPES)
            trials_types = set(ALL_EXCLUSION_TRIALS_TYPES)

        elif "cf saving" in filename_remainder:
            return None

        elif "bulk reprocessing" in filename_remainder:
            return None

        elif "map generation" in filename_remainder:
            return None

        else:
            logger.warning(f"Unknown exclusion type: {path.absolute}")
            return None

        with path.open() as f:
            return cls.from_file(f, data_types, trials_types)
