#!/usr/bin/env python3

import pandas as pd
import kagglehub
import shutil
import os


def main():
    print("Downloading dataset...")

    # Download dataset to cache
    path = kagglehub.dataset_download("firecastrl/us-wildfire-dataset")

    # Current working directory
    dest_dir = os.getcwd()

    print(f"Copying files to: {dest_dir}")

    # Copy files to working directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir)
            print(f"Moved {filename} to {dest_dir}")

    print("Done.")


if __name__ == "__main__":
    main()