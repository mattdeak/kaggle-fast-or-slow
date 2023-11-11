from typing import Any

import numpy as np
import numpy.typing as npt

# These groups are meant to distinctly allocate opcodes into groups that are similar to each other.
NUM_OPCODES = 121
DEFAULT_GROUPS = {
    0: [2, 95, 59, 32, 60, 1, 66, 73, 50, 51, 37, 38, 94, 81, 103, 90],
    1: [7, 30, 91, 96, 15, 40, 80, 118, 107],
    2: [53, 52, 54, 55, 87, 88, 89, 11, 12, 65, 19],
    3: [20, 83, 17, 49],
    4: [34, 26, 16, 99],
    5: [27, 29, 28, 35, 36, 42, 82, 22, 92, 108, 75, 76, 98, 62, 93],
    6: [48, 77, 79, 78, 13, 21, 67, 46, 25, 12, 119],
    7: [70, 72, 71, 111, 5, 109, 110, 10, 9, 8, 57, 58, 104, 112, 113],
    8: [14, 23, 102],
    9: [
        85,
        86,
        68,
        69,
        47,
        61,
        6,
        18,
        105,
        106,
        104,
        112,
        113,
        5,
        109,
        110,
        111,
        84,
    ],
    10: [4, 100, 45, 44, 43, 63, 64, 74, 33],
    11: [115, 116, 117, 114],
    12: [63, 24],
    13: [31],
    14: [41],
}


class OpcodeGroupOHEEmbedder:
    def __init__(
        self, groups: dict[int, list[int]] = DEFAULT_GROUPS, normalize: bool = True
    ):
        self.groups = groups
        self.normalize = normalize

    def __call__(self, opcodes: npt.NDArray[Any]) -> npt.NDArray[Any]:
        group_opcodes = np.zeros((opcodes.shape[0], len(self.groups)))
        for group_num, group in self.groups.items():
            group_opcodes[np.isin(opcodes, group), group_num] = 1

        if self.normalize:
            group_opcodes = group_opcodes / np.sum(group_opcodes, axis=1, keepdims=True)

        return group_opcodes


class OpcodeOHEEmbedder:
    def __init__(self, ohe_drop_threshold: float = 0.05) -> None:
        self._fitted = False
        self.ohe_drop_threshold = ohe_drop_threshold

    def fit(self, opcodes: npt.NDArray[Any]) -> None:
        opcode_counts = np.zeros(NUM_OPCODES)
        for opcode in opcodes:
            opcode_counts[opcode] += 1

        self._opcode_counts = opcode_counts
        self.drop_mask = opcode_counts < self.ohe_drop_threshold * np.sum(opcode_counts)
        self._fitted = True

    def __call__(self, opcodes: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if not self._fitted:
            raise RuntimeError("OpcodeOHEEmbedder not fitted")

        ohe_opcodes = np.zeros((opcodes.shape[0], NUM_OPCODES))
        ohe_opcodes[np.arange(opcodes.shape[0]), opcodes] = 1

        ohe_opcodes = ohe_opcodes[:, ~self.drop_mask]

        return ohe_opcodes
