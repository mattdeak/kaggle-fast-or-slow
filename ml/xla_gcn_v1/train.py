from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
import pytorch

from lib.metrics import xla_slowdown_from_runtime_preds
from lib.transforms.node_sum_pooling_with_graph_features_v1 import get_data
