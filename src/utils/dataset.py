import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.utils.preprocessing import get_transform


class PokemonDataset(Dataset):
    """PyTorch Dataset for Pokemon images and their primary type labels."""

    def __init__(self, data_dir, transform=None):
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
