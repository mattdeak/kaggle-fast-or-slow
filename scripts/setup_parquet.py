import argparse
import os
from typing import Literal

from tqdm.auto import tqdm

from lib.parquet_writer import ParquetWriter

Split = Literal["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="data/npz/tile/xla/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])

    return parser.parse_args()


def setup_parquet(
    directory: str,
    splits: list[Split] = [
        "train",
        "valid",
        "test",
    ],
    verbose: bool = True,
) -> None:
    for split in splits:
        if verbose:
            print("Processing", split)
        src_path = os.path.join(directory, split)
        outpath = os.path.join("data/parquet/", split)
        writer = ParquetWriter(src_path, outpath)
        files = os.listdir(src_path)

        for file in tqdm(files, disable=not verbose):
            writer.process_file(file)


if __name__ == "__main__":
    args = parse_args()
    setup_parquet(args.directory, args.splits, args.verbose)
