import os

import numpy as np


def main():
    processed_dir = os.path.join(os.getcwd(), "data/processed")
    subdirs = os.listdir(processed_dir)
    for subdir in subdirs:
        model_files = os.listdir(os.path.join(processed_dir, subdir))
        for model_file in model_files:
            print("Checking {}".format(os.path.join(processed_dir, subdir, model_file)))
            numpy_files = os.listdir(os.path.join(processed_dir, subdir, model_file))
            for numpy_file in numpy_files:
                if not numpy_file.endswith(".npy"):
                    continue

                numpy_file_path = os.path.join(
                    processed_dir, subdir, model_file, numpy_file
                )
                data = np.load(numpy_file_path)

                # check for nans, infs
                if np.isnan(data).any():
                    print("Nans in {}".format(numpy_file_path))

                if np.isinf(data).any():
                    print("Infs in {}".format(numpy_file_path))

                # check for neg infs
                if np.isneginf(data).any():
                    print("Neg Infs in {}".format(numpy_file_path))

    for file in os.listdir(processed_dir):
        if file.endswith(".csv"):
            print(file)


if __name__ == "__main__":
    main()
