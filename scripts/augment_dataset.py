"""
Offline data augmentation script for the Pokemon type classifier.

Generates augmented image variants and saves them to disk, creating a
self-contained dataset directory that can be used directly with --data-dir.

Why offline augmentation:
  Online augmentation (applied each epoch) regularises training but doesn't
  increase dataset size. Offline augmentation physically multiplies the number
  of images, giving the model genuinely more data points per class and making
  early stopping more reliable on the small validation set.

Usage:
    python scripts/augment_dataset.py
    python scripts/augment_dataset.py --target-per-class 200
    python scripts/augment_dataset.py --output-dir data/augmented --target-per-class 150

Then train on the augmented dataset:
    python classifiers/pytorch/train.py --data-dir data/augmented
"""

import argparse
import os
import random
import shutil
import sys
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocessing import get_offline_augmentation, RGBAToRGB


def parse_args():
    p = argparse.ArgumentParser(description="Augment Pokemon dataset offline")
    p.add_argument(
        "--data-dir",
        default=os.path.join(PROJECT_ROOT, "data"),
        help="Path to the original dataset directory (must contain pokemon.csv and images/)",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "data", "augmented"),
        help="Where to write the augmented dataset (default: data/augmented/)",
    )
    p.add_argument(
        "--target-per-class",
        type=int,
        default=200,
        help="Target total images per class (original + augmented). Default: 200",
    )
    p.add_argument(
        "--max-aug-per-image",
        type=int,
        default=20,
        help="Hard cap on augmented variants generated from a single source image. Default: 20",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists",
    )
    return p.parse_args()


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB with a white background."""
    return RGBAToRGB()(img)


def augment_and_save(
    source_path: str,
    base_name: str,
    count: int,
    out_images_dir: str,
    aug_transform,
    rows: list,
    type1: str,
    type2: str,
):
    """
    Generate `count` augmented variants of the image at `source_path`.
    Saves each as {base_name}_aug_{i:03d}.png and appends to `rows`.
    """
    try:
        img = Image.open(source_path)
        img = to_rgb(img)
    except Exception as e:
        print(f"  [WARN] Could not open {source_path}: {e}")
        return

    for i in range(count):
        aug_img = aug_transform(img)
        aug_name = f"{base_name}_aug_{i:03d}"
        out_path = os.path.join(out_images_dir, f"{aug_name}.png")
        aug_img.save(out_path, format="PNG")
        rows.append({"Name": aug_name, "Type1": type1, "Type2": type2})


def main():
    args = parse_args()
    random.seed(args.seed)

    # ── Validate input ──────────────────────────────────────────────────────
    csv_path = os.path.join(args.data_dir, "pokemon.csv")
    images_dir = os.path.join(args.data_dir, "images")
    if not os.path.exists(csv_path):
        print(f"[ERROR] pokemon.csv not found at {csv_path}")
        sys.exit(1)
    if not os.path.exists(images_dir):
        print(f"[ERROR] images/ directory not found at {images_dir}")
        sys.exit(1)

    # ── Prepare output directory ─────────────────────────────────────────────
    out_images_dir = os.path.join(args.output_dir, "images")
    out_csv_path = os.path.join(args.output_dir, "pokemon.csv")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            shutil.rmtree(args.output_dir)
            print(f"[INFO] Removed existing output directory: {args.output_dir}")
        else:
            print(
                f"[ERROR] Output directory already exists: {args.output_dir}\n"
                f"        Use --overwrite to replace it."
            )
            sys.exit(1)

    os.makedirs(out_images_dir, exist_ok=True)

    # ── Load CSV, filter to images that exist ────────────────────────────────
    df = pd.read_csv(csv_path)
    valid_rows = []
    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, f"{row['Name']}.png")
        if os.path.exists(img_path):
            valid_rows.append(row)
    df = pd.DataFrame(valid_rows).reset_index(drop=True)

    # Compute per-class counts (by Type1)
    class_counts = Counter(df["Type1"].tolist())
    print(f"\n[INFO] Original dataset: {len(df)} images across {len(class_counts)} classes")
    print(f"[INFO] Target per class: {args.target_per_class}")
    print(f"[INFO] Class counts (original):")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: x[1]):
        needed = max(0, args.target_per_class - cnt)
        print(f"       {cls:<12} {cnt:>3} images  →  need {needed:>3} augmented")

    # ── Copy original images and build output rows ───────────────────────────
    print(f"\n[STEP 1] Copying {len(df)} original images...")
    output_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  copying"):
        src = os.path.join(images_dir, f"{row['Name']}.png")
        dst = os.path.join(out_images_dir, f"{row['Name']}.png")
        # Save as RGB PNG (normalise RGBA along the way)
        img = to_rgb(Image.open(src))
        img.save(dst, format="PNG")
        output_rows.append({
            "Name": row["Name"],
            "Type1": row["Type1"],
            "Type2": row.get("Type2", ""),
        })

    # ── Generate augmented images ────────────────────────────────────────────
    aug_transform = get_offline_augmentation()
    print(f"\n[STEP 2] Generating augmented images...")

    # Group rows by Type1 class
    class_rows = {}
    for _, row in df.iterrows():
        cls = row["Type1"]
        class_rows.setdefault(cls, []).append(row)

    total_augmented = 0
    for cls, rows in sorted(class_rows.items()):
        current_count = len(rows)
        needed = max(0, args.target_per_class - current_count)
        if needed == 0:
            print(f"  {cls:<12} already has {current_count} images — skipping augmentation")
            continue

        # Distribute `needed` augmentations across the source images
        # Cycle through source images so each gets a roughly equal share
        aug_per_image = min(args.max_aug_per_image, -(-needed // len(rows)))  # ceiling div
        generated = 0

        for row in rows:
            if generated >= needed:
                break
            to_gen = min(aug_per_image, needed - generated)
            augment_and_save(
                source_path=os.path.join(images_dir, f"{row['Name']}.png"),
                base_name=row["Name"],
                count=to_gen,
                out_images_dir=out_images_dir,
                aug_transform=aug_transform,
                rows=output_rows,
                type1=row["Type1"],
                type2=row.get("Type2", ""),
            )
            generated += to_gen

        total_augmented += generated
        print(f"  {cls:<12} {current_count:>3} → {current_count + generated:>3} (+{generated})")

    # ── Write combined CSV ───────────────────────────────────────────────────
    out_df = pd.DataFrame(output_rows)
    # Ensure Type2 column stores NaN (not empty string) for compatibility with PokemonDataset
    out_df["Type2"] = out_df["Type2"].replace("", float("nan"))
    out_df.to_csv(out_csv_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    final_counts = Counter(out_df["Type1"].tolist())
    print(f"\n[DONE] Augmented dataset written to: {args.output_dir}")
    print(f"       Original images : {len(df)}")
    print(f"       Augmented images: {total_augmented}")
    print(f"       Total images    : {len(out_df)}")
    print(f"       Classes         : {len(final_counts)}")
    print(f"       Min per class   : {min(final_counts.values())}")
    print(f"       Max per class   : {max(final_counts.values())}")
    print(f"\nTo train on the augmented dataset:")
    print(f"  python classifiers/pytorch/train.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
