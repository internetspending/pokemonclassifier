"""
Fixed Pokemon Dataset that loads ALL images from type folders.

This version loads images directly from the type-organized folder structure:
  data/images/Electric/*.png
  data/images/Fire/*.png
  etc.

Instead of relying on the CSV to list specific Pokemon names.
"""

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from utils.preprocessing import get_transform


class PokemonDataset(Dataset):
    """PyTorch Dataset for Pokemon images organized by type folders."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or get_transform()
        
        # Path to images directory
        images_dir = Path(data_dir) / "images"
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all type folders (Bug, Water, Fire, etc.)
        type_folders = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        
        if not type_folders:
            raise ValueError(f"No type folders found in {images_dir}")
        
        # Build label mapping from folder names
        self.label_names = [folder.name for folder in type_folders]
        self._label_to_idx = {name: i for i, name in enumerate(self.label_names)}
        
        # Load all images from all type folders
        self.samples = []
        for type_folder in type_folders:
            type_name = type_folder.name
            type_idx = self._label_to_idx[type_name]
            
            # Get all image files in this type folder
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                for img_path in type_folder.glob(ext):
                    self.samples.append((str(img_path), type_idx))
        
        print(f"[PokemonDataset] Loaded {len(self.samples)} images across {len(self.label_names)} types")
        
        # Print distribution
        from collections import Counter
        type_counts = Counter([label for _, label in self.samples])
        print(f"[PokemonDataset] Images per type:")
        for type_name in self.label_names:
            type_idx = self._label_to_idx[type_name]
            count = type_counts[type_idx]
            print(f"  {type_name:12} {count:4} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label


# Backward compatibility: keep old CSV-based loading as option
class PokemonDatasetCSV(Dataset):
    """Original CSV-based dataset (only loads Pokemon listed in CSV)."""

    def __init__(self, data_dir, transform=None):
        import pandas as pd
        
        self.data_dir = data_dir
        self.transform = transform or get_transform()

        df = pd.read_csv(os.path.join(data_dir, "pokemon.csv"))

        # Build label mapping from unique Type1 values
        self.label_names = sorted(df["Type1"].unique().tolist())
        self._label_to_idx = {name: i for i, name in enumerate(self.label_names)}

        # Keep only rows with a valid image file
        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(data_dir, "images", f"{row['Name']}.png")
            if os.path.exists(img_path):
                self.samples.append((img_path, self._label_to_idx[row["Type1"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label