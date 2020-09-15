import abc
import typing as t

import nptyping as npt


class BaseDataProcessor(abc.ABC):
    """Base class for an object that can take the output array from a trial and modify it in some
    way, returning a new array of equal size and meaning"""

    @abc.abstractmethod
    def transform(self, array: npt.NDArray[(t.Any, t.Any), t.Any]) -> npt.NDArray[(t.Any, t.Any), t.Any]:
        ...  # pragma: no cover
