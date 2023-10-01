import argparse
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Literal

from tqdm.auto import tqdm

from lib.parquet_writer import ParquetWriter

Split = Literal["train", "valid", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", default="data/npz/tile/xla/")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])

    return parser.parse_args()


@dataclass
class WorkerArgs:
    src_path: str
    outpath: str
    files: list[str]
    quiet: bool


def process_files(args: WorkerArgs) -> None:
    writer = ParquetWriter(args.src_path, args.outpath)

    for file in tqdm(args.files, disable=args.quiet):
        writer.process_file(file)


def chunk_files(files: list[str], num_chunks: int):
    chunk_size = len(files) // num_chunks
    return [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]


def setup_parquet(
    directory: str,
    splits: list[str] = ["train", "valid", "test"],
    quiet: bool = True,
) -> None:
    num_processes = min(
        cpu_count(), len(splits)
    )  # Ensure we don't create unnecessary processes

    for split in splits:
        if not quiet:
            print("Processing", split)

        src_path = os.path.join(directory, split)
        outpath = os.path.join("data/parquet/", split)
        files = os.listdir(src_path)

        with Pool(processes=num_processes) as pool:
            pool.map(
                process_files,
                [
                    WorkerArgs(src_path, outpath, chunk, quiet)
                    for chunk in chunk_files(files, num_processes)
                ],
            )


def main() -> None:
    args = parse_args()
    setup_parquet(args.directory, args.splits, args.quiet)


if __name__ == "__main__":
    main()
