"""
EfficientNet-B0 inference smoke test.

Usage:
    python classifiers/pytorch/smoke_test.py                             # dummy forward pass
    python classifiers/pytorch/smoke_test.py data/images/bulbasaur.png   # inference on image
    python classifiers/pytorch/smoke_test.py --checkpoint models/best_pokemon_classifier.pt data/images/bulbasaur.png
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 18


def parse_args():
    p = argparse.ArgumentParser(description="EfficientNet-B0 smoke test")
    p.add_argument("image", nargs="?", default=None, help="Path to an image file")
    p.add_argument("--checkpoint", default=None, help="Path to a trained .pt checkpoint")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] GPU = {torch.cuda.get_device_name(0)}")

    # Load pretrained weights + preprocessing
    weights = EfficientNet_B0_Weights.DEFAULT
    preprocess = weights.transforms()

    # Build model
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    # Load trained checkpoint if provided
    label_names = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        label_names = ckpt.get("label_names")
        print(f"[OK] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[INFO] No checkpoint loaded (predictions will be random)")

    model = model.to(device).eval()

    # Run inference
    if args.image:
        print(f"[RUN] Image inference on: {args.image}")
        img = Image.open(args.image).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)
    else:
        print("[RUN] Dummy forward pass (no image provided)")
        x = torch.randn(1, 3, 224, 224, device=device)

    print(f"[INFO] input shape = {tuple(x.shape)}")

    with torch.inference_mode():
        logits = model(x)

    print(f"[RESULT] output shape = {tuple(logits.shape)} (expected: (1, {NUM_CLASSES}))")

    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(5, NUM_CLASSES))
    print("[RESULT] top-5 predictions:")
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        name = label_names[idx] if label_names else str(idx)
        print(f"  {name:>10}  {p:.4f}")


if __name__ == "__main__":
    main()
