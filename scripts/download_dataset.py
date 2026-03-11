"""
Download the Pokemon Images and Types dataset from Kaggle using kagglehub.

Usage:
    python scripts/download_dataset.py

Requires:
    pip install kagglehub
    Kaggle credentials (set KAGGLE_USERNAME and KAGGLE_KEY env vars,
    or place kaggle.json in ~/.kaggle/)
"""

import os
import shutil
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import kagglehub
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATASET = "vishalsubbiah/pokemon-images-and-types"


def main():
    csv_path = os.path.join(DATA_DIR, "pokemon.csv")
    images_path = os.path.join(DATA_DIR, "images")

    if os.path.exists(csv_path) and os.path.exists(images_path):
        print("Dataset already exists at:", DATA_DIR)
        print("Skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    cached_path = kagglehub.dataset_download(DATASET)
    print(f"Downloaded to cache: {cached_path}")

    # Copy pokemon.csv
    src_csv = os.path.join(cached_path, "pokemon.csv")
    dst_csv = os.path.join(DATA_DIR, "pokemon.csv")
    shutil.copy2(src_csv, dst_csv)
    print(f"Copied pokemon.csv -> {dst_csv}")

    # Copy images/ directory
    src_images = os.path.join(cached_path, "images")
    dst_images = os.path.join(DATA_DIR, "images")
    if os.path.exists(dst_images):
        shutil.rmtree(dst_images)
    shutil.copytree(src_images, dst_images)
    image_count = len([f for f in os.listdir(dst_images) if f.endswith(".png")])
    print(f"Copied {image_count} images -> {dst_images}")

    print("\nDataset ready at:", DATA_DIR)


if __name__ == "__main__":
    main()