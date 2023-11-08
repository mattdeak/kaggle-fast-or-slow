from typing import Any

import numpy as np
import numpy.typing as npt


class LogTargetTransform:
    def __call__(self, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.log(y + 1)
