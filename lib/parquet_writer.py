import os
from typing import Any

import numpy as np
import polars as pl

from lib.feature_name_mapping import (CONFIG_FEATURE_MAP, NODE_FEATURE_MAP,
                                      OPCODE_MAP)


class ParquetWriter:
    """Process the npz files into parquet files for easier work"""

    def __init__(self, source_dir: str, output_dir: str) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir

    def node_filepath(self, filename: str) -> str:
        os.makedirs(os.path.join(self.output_dir, "node"), exist_ok=True)
        return os.path.join(self.output_dir, "node", f"{filename}.parquet")

    def config_filepath(self, filename: str) -> str:
        os.makedirs(os.path.join(self.output_dir, "config"), exist_ok=True)
        return os.path.join(self.output_dir, "config", f"{filename}.parquet")

    def edge_filepath(self, filename: str) -> str:
        os.makedirs(os.path.join(self.output_dir, "edge"), exist_ok=True)
        return os.path.join(self.output_dir, "edge", f"{filename}.parquet")

    def process_file(self, file: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        d: dict[str, Any] = dict(np.load(os.path.join(self.source_dir, file)))
        d["file"] = file

        self.process_node_data(d)
        self.process_config_data(d)
        self.process_edge_data(d)

    def process_node_data(self, data: dict[str, Any]) -> None:
        """Process the node features"""
        node_feat = {
            NODE_FEATURE_MAP[j]: data["node_feat"][:, j]
            for j in range(data["node_feat"].shape[1])
        }

        opcodes = [OPCODE_MAP[opc] for opc in data["node_opcode"]]  # type: ignore
        node_df = pl.DataFrame(
            {
                **node_feat,
                "node_opcode": opcodes,
                "node_id": np.arange(0, len(opcodes)),
                "file_id": data["file"],
            }
        )

        node_df.write_parquet(self.node_filepath(data["file"]))

    def process_config_data(self, data: dict[str, Any]) -> None:
        config_feat = {
            CONFIG_FEATURE_MAP[j]: data["config_feat"][:, j]
            for j in range(data["config_feat"].shape[1])
        }

        config_df = pl.DataFrame(
            {
                **config_feat,
                "config_runtime": data["config_runtime"],
                "config_runtime_normalizers": data["config_runtime_normalizers"],
                "file_id": data["file"],
            }
        )
        config_df.write_parquet((self.config_filepath(data["file"])))

    def process_edge_data(self, data: dict[str, Any]) -> None:
        edge_df = pl.DataFrame(
            {
                "from": data["edge_index"][:, 0],
                "to": data["edge_index"][:, 1],
                "file_id": data["file"],
            }
        )

        edge_df.write_parquet(self.edge_filepath(data["file"]))
