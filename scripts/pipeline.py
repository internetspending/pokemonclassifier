"""
Pipeline script that ties together preprocessing, dataset loading,
and the EfficientNet-B0 smoke test into a single end-to-end run.

Usage:
    python -m scripts.pipeline          (from project root)
    python scripts/pipeline.py          (from project root)
"""

import os
import sys

# Ensure project root is on sys.path so `utils.*` imports resolve
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from utils.preprocessing import get_transform
from utils.dataset import PokemonDataset

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
NUM_CLASSES = 18


def step_preprocessing():
    """Verify that the preprocessing transform pipeline builds correctly."""
    print("=" * 60)
    print("STEP 1: Preprocessing pipeline")
    print("=" * 60)

    transform = get_transform()
    print(f"[OK] Transform pipeline built successfully:\n{transform}\n")
    return transform


def step_dataset():
    """Load PokemonDataset and verify a sample batch through the DataLoader."""
    print("=" * 60)
    print("STEP 2: Dataset loading")
    print("=" * 60)

    dataset = PokemonDataset(data_dir=DATA_DIR)
    print(f"[OK] Loaded {len(dataset)} samples, "
          f"{len(dataset.label_names)} classes")
    print(f"[OK] Classes: {dataset.label_names}")

    # Single-sample check
    img, label = dataset[0]
    print(f"[OK] Sample 0 — shape: {tuple(img.shape)}, "
          f"label: {label} ({dataset.label_names[label]})")

    # DataLoader batch check
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch_imgs, batch_labels = next(iter(loader))
    print(f"[OK] DataLoader batch — images: {tuple(batch_imgs.shape)}, "
          f"labels: {tuple(batch_labels.shape)}")

    return dataset


def step_smoke_test(dataset):
    """Build EfficientNet-B0 and run inference on a real dataset sample."""
    print("=" * 60)
    print("STEP 3: Model smoke test (EfficientNet-B0)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    if device == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Load pretrained weights and build model
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Replace classifier head (1000 -> 18)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    print(f"[OK] Classifier head replaced: 1000 -> {NUM_CLASSES}")

    model = model.to(device).eval()

    # Inference on a real sample from the dataset
    img, label = dataset[0]
    x = img.unsqueeze(0).to(device)
    print(f"[INFO] Input shape: {tuple(x.shape)}")

    with torch.inference_mode():
        logits = model(x)

    print(f"[OK] Output shape: {tuple(logits.shape)}")

    # Top-5 predictions
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(5, NUM_CLASSES))
    true_name = dataset.label_names[label]
    print(f"[OK] Top-5 predictions (true type: {true_name}):")
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"     {dataset.label_names[idx]:>10}  {p:.4f}")


def main():
    print("Pokemon Classifier — Full Pipeline\n")

    step_preprocessing()
    print()
    dataset = step_dataset()
    print()
    step_smoke_test(dataset)

    print()
    print("=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
