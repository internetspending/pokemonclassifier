import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from utils.preprocessing import get_transform


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
                type1_idx = self._label_to_idx[row["Type1"]]
                type2_raw = row.get("Type2")
                type2_idx = (
                    self._label_to_idx[type2_raw]
                    if pd.notna(type2_raw) and type2_raw in self._label_to_idx
                    else None
                )
                self.samples.append((img_path, type1_idx, type2_idx))

    def __len__(self):
        return len(self.samples)

    def valid_labels(self, idx):
        """Return the set of valid type indices for sample `idx` (type1 always, type2 when present)."""
        _, type1_idx, type2_idx = self.samples[idx]
        labels = {type1_idx}
        if type2_idx is not None:
            labels.add(type2_idx)
        return labels

    def __getitem__(self, idx):
        img_path, type1_idx, _ = self.samples[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, type1_idx
