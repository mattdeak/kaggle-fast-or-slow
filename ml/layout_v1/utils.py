from typing import Any, TypeVar

from numpy.typing import NDArray
from torch import Tensor

T = TypeVar("T", Tensor, NDArray[Any])


def get_rank(x: T) -> T:
    """Get the rank of each element in x.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Rank of each element in x.
    """
    return x.argsort().argsort()
