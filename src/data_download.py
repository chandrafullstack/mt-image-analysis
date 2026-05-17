"""Download public mitochondria EM datasets."""
import os
import urllib.request
from pathlib import Path

DATASETS = {
    "epfl_train": {
        "url": "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training.tif",
        "dest": "data/raw/epfl_rat/training.tif",
    },
    "epfl_gt": {
        "url": "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/training_groundtruth.tif",
        "dest": "data/raw/epfl_rat/training_groundtruth.tif",
    },
}


def download_datasets(datasets: dict = DATASETS):
    for name, info in datasets.items():
        dest = Path(info["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(info["url"], dest)
            print(f"  Saved to {dest}")
        else:
            print(f"  {name} already exists, skipping.")


if __name__ == "__main__":
    download_datasets()
