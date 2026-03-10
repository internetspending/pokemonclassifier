"""
Unseen Pokemon Testing - Test if model learns TYPE features vs memorizing

This script creates a special train/test split where:
  - Training: Uses ALL images of most Pokemon (full training set, minus holdouts)
  - test_unseen: Uses ONLY Pokemon never seen during training

This tests if the model learned visual patterns for types (e.g., "Fire types are red/orange")
vs just memorizing specific Pokemon.

KEY FIX: All seen Pokemon images go to train (no inner 80/20 split).
         This keeps the training set as large as the original dataset.

Usage:
    # Hold out ~1-2 Pokemon per type automatically (low ratio recommended)
    python test_unseen_pokemon.py --holdout-ratio 0.15

    # Hold out specific Pokemon by name
    python test_unseen_pokemon.py --holdout-pokemon "Electabuzz,Gyarados,Dragonite,Arcanine"
"""

import argparse
import os
import json
import random
from pathlib import Path
from collections import defaultdict
import shutil


def parse_args():
    p = argparse.ArgumentParser(description="Create unseen Pokemon test set")
    p.add_argument("--data-dir", default="data/images",
                   help="Directory with type folders (e.g. data/images/Fire/*.png)")
    p.add_argument("--output-dir", default="data_unseen_pokemon",
                   help="Output directory for split dataset")
    p.add_argument("--holdout-ratio", type=float, default=0.15,
                   help="Ratio of Pokemon to hold out per type (default: 0.15 = ~1-2 per type)")
    p.add_argument("--holdout-pokemon", default=None,
                   help="Comma-separated list of specific Pokemon names to hold out")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--min-images-per-pokemon", type=int, default=3,
                   help="Minimum images a Pokemon must have to be held out")
    return p.parse_args()


def extract_pokemon_name(filename):
    """Extract Pokemon name from filename.
    Handles: pikachu_1.png, pikachu_gen1.png, charizard_official.png, etc.
    """
    name = Path(filename).stem

    # Remove known suffixes
    suffixes_to_remove = [
        '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9',
        '_gen1', '_official', '_card', '_shiny', '_mega'
    ]
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:name.rfind(suffix)]

    # Remove trailing digits
    while name and name[-1].isdigit():
        name = name[:-1]

    return name.strip('_- ').lower()


def group_images_by_pokemon(type_folder):
    """Group image paths by extracted Pokemon name."""
    pokemon_groups = defaultdict(list)
    for img_file in type_folder.iterdir():
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            pokemon_name = extract_pokemon_name(img_file.name)
            pokemon_groups[pokemon_name].append(img_file)
    return pokemon_groups


def select_holdout_pokemon(pokemon_groups, holdout_ratio, min_images, seed):
    """Randomly select which Pokemon to hold out, only picking ones with enough images."""
    random.seed(seed)

    valid_pokemon = {
        name: imgs
        for name, imgs in pokemon_groups.items()
        if len(imgs) >= min_images
    }

    if not valid_pokemon:
        return set()

    num_holdout = max(1, int(len(valid_pokemon) * holdout_ratio))
    holdout = set(random.sample(list(valid_pokemon.keys()), num_holdout))
    return holdout


def create_unseen_pokemon_split(data_dir, output_dir, holdout_ratio,
                                holdout_pokemon_list, min_images, seed):
    """
    Create the split:
      - train/          <- ALL images of seen Pokemon  (full training volume!)
      - test_unseen/    <- ALL images of held-out Pokemon  (never seen during training)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    train_dir = output_dir / "train"
    test_unseen_dir = output_dir / "test_unseen"

    for d in [train_dir, test_unseen_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {
        'types': {},
        'total_pokemon': 0,
        'holdout_pokemon': 0,
        'train_images': 0,
        'test_unseen_images': 0,
        'holdout_pokemon_names': []
    }

    type_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for type_folder in type_folders:
        type_name = type_folder.name
        print(f"\n📁 Processing {type_name}...")

        (train_dir / type_name).mkdir(exist_ok=True)
        (test_unseen_dir / type_name).mkdir(exist_ok=True)

        pokemon_groups = group_images_by_pokemon(type_folder)
        print(f"   Found {len(pokemon_groups)} unique Pokemon")

        # Decide which Pokemon to hold out
        if holdout_pokemon_list:
            holdout_pokemon = set(
                name.lower().strip()
                for name in holdout_pokemon_list
                if name.lower().strip() in pokemon_groups
            )
        else:
            holdout_pokemon = select_holdout_pokemon(
                pokemon_groups, holdout_ratio, min_images, seed
            )

        print(f"   Holding out {len(holdout_pokemon)} Pokemon for test_unseen:")
        for name in sorted(holdout_pokemon):
            print(f"      - {name.capitalize()} ({len(pokemon_groups[name])} images)")

        type_stats = {
            'total_pokemon': len(pokemon_groups),
            'holdout_pokemon': len(holdout_pokemon),
            'train_images': 0,
            'test_unseen_images': 0,
        }

        for pokemon_name, images in pokemon_groups.items():
            if pokemon_name in holdout_pokemon:
                # ---- UNSEEN: all images go to test_unseen ----
                for img in images:
                    shutil.copy2(img, test_unseen_dir / type_name / img.name)
                    type_stats['test_unseen_images'] += 1
                stats['holdout_pokemon_names'].append(f"{pokemon_name} ({type_name})")
            else:
                # ---- SEEN: ALL images go to train (no inner split!) ----
                for img in images:
                    shutil.copy2(img, train_dir / type_name / img.name)
                    type_stats['train_images'] += 1

        stats['types'][type_name] = type_stats
        stats['total_pokemon'] += type_stats['total_pokemon']
        stats['holdout_pokemon'] += type_stats['holdout_pokemon']
        stats['train_images'] += type_stats['train_images']
        stats['test_unseen_images'] += type_stats['test_unseen_images']

    return stats


def main():
    args = parse_args()

    print("=" * 80)
    print("UNSEEN POKEMON TESTING - DATASET SPLIT")
    print("=" * 80)
    print(f"Source:         {args.data_dir}")
    print(f"Output:         {args.output_dir}")
    print(f"Holdout ratio:  {args.holdout_ratio * 100:.0f}% of Pokemon per type")
    if args.holdout_pokemon:
        print(f"Specific Pokemon: {args.holdout_pokemon}")
    print()

    holdout_pokemon_list = None
    if args.holdout_pokemon:
        holdout_pokemon_list = [n.strip() for n in args.holdout_pokemon.split(',')]

    print("Creating unseen Pokemon split...")
    print("-" * 80)

    stats = create_unseen_pokemon_split(
        args.data_dir,
        args.output_dir,
        args.holdout_ratio,
        holdout_pokemon_list,
        args.min_images_per_pokemon,
        args.seed
    )

    # Save stats
    stats_file = Path(args.output_dir) / "split_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("SPLIT COMPLETE")
    print("=" * 80)
    print(f"Total Pokemon:        {stats['total_pokemon']}")
    print(f"Held-out Pokemon:     {stats['holdout_pokemon']}")
    print()
    print(f"Training set:         {stats['train_images']:5} images  ← full volume, seen Pokemon only")
    print(f"Test (UNSEEN):        {stats['test_unseen_images']:5} images  ← never seen during training!")
    print()
    print("Held-out Pokemon (generalization test targets):")
    for name in sorted(stats['holdout_pokemon_names'])[:30]:
        print(f"  - {name}")
    if len(stats['holdout_pokemon_names']) > 30:
        print(f"  ... and {len(stats['holdout_pokemon_names']) - 30} more")

    print()
    print(f"Stats saved to: {stats_file}")
    print()
    print("=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Train on the split dataset (same volume as normal training!):")
    print(f"   python classifiers/pytorch/train.py --data-dir {args.output_dir}/train")
    print()
    print("2. Evaluate on UNSEEN Pokemon (the key test!):")
    print(f"   python evaluate_unseen.py --checkpoint models/best_pokemon_classifier.pt \\")
    print(f"       --test-dir {args.output_dir}/test_unseen")
    print()
    print("Interpreting results:")
    print("  High unseen accuracy  → model LEARNED visual type features ✓")
    print("  Low unseen accuracy   → model MEMORIZED specific Pokemon  ✗")
    print("=" * 80)


if __name__ == "__main__":
    main()