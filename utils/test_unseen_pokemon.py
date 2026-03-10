"""
Unseen Pokemon Testing - Test if model learns TYPE features vs memorizing

This script creates a special train/test split where:
  - Training: Uses most Pokemon
  - Testing: Uses ONLY Pokemon never seen during training

This tests if the model learned visual patterns for types (e.g., "Fire types are red/orange")
vs just memorizing specific Pokemon.

Usage:
    python test_unseen_pokemon.py --holdout-ratio 0.2
    python test_unseen_pokemon.py --holdout-pokemon "Electabuzz,Gyarados,Dragonite"
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
                   help="Directory with type folders")
    p.add_argument("--output-dir", default="data_unseen_pokemon",
                   help="Output directory for split dataset")
    p.add_argument("--holdout-ratio", type=float, default=0.2,
                   help="Ratio of Pokemon to hold out per type (default: 0.2 = 20%%)")
    p.add_argument("--holdout-pokemon", default=None,
                   help="Comma-separated list of specific Pokemon to hold out")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--min-images-per-pokemon", type=int, default=3,
                   help="Minimum images a Pokemon must have to be included")
    return p.parse_args()


def extract_pokemon_name(filename):
    """Extract Pokemon name from filename (handles pikachu_1.png, pikachu_gen1.png, etc.)"""
    # Remove extension
    name = Path(filename).stem
    
    # Remove common suffixes
    suffixes_to_remove = ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9',
                          '_gen1', '_official', '_card', '_shiny', '_mega']
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:name.rfind(suffix)]
    
    # Remove numbers at end
    while name and name[-1].isdigit():
        name = name[:-1]
    
    # Clean up
    name = name.strip('_- ')
    
    return name.lower()


def group_images_by_pokemon(type_folder):
    """Group images by Pokemon name"""
    pokemon_groups = defaultdict(list)
    
    for img_file in type_folder.iterdir():
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            pokemon_name = extract_pokemon_name(img_file.name)
            pokemon_groups[pokemon_name].append(img_file)
    
    return pokemon_groups


def select_holdout_pokemon(pokemon_groups, holdout_ratio, min_images, seed):
    """Select which Pokemon to hold out from training"""
    random.seed(seed)
    
    # Only consider Pokemon with enough images
    valid_pokemon = {
        name: images 
        for name, images in pokemon_groups.items() 
        if len(images) >= min_images
    }
    
    if not valid_pokemon:
        return set()
    
    # Calculate how many to hold out
    num_holdout = max(1, int(len(valid_pokemon) * holdout_ratio))
    
    # Randomly select Pokemon to hold out
    pokemon_names = list(valid_pokemon.keys())
    holdout_pokemon = set(random.sample(pokemon_names, num_holdout))
    
    return holdout_pokemon


def create_unseen_pokemon_split(data_dir, output_dir, holdout_ratio, 
                                holdout_pokemon_list, min_images, seed):
    """Create train/test split with unseen Pokemon in test set"""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_dir = output_dir / "train"
    test_unseen_dir = output_dir / "test_unseen"
    test_seen_dir = output_dir / "test_seen"
    
    for dir_path in [train_dir, test_unseen_dir, test_seen_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        'types': {},
        'total_pokemon': 0,
        'holdout_pokemon': 0,
        'train_images': 0,
        'test_unseen_images': 0,
        'test_seen_images': 0,
        'holdout_pokemon_names': []
    }
    
    # Process each type
    type_folders = [d for d in data_dir.iterdir() if d.is_dir()]
    
    for type_folder in sorted(type_folders):
        type_name = type_folder.name
        print(f"\n📁 Processing {type_name}...")
        
        # Create type folders in output
        (train_dir / type_name).mkdir(exist_ok=True)
        (test_unseen_dir / type_name).mkdir(exist_ok=True)
        (test_seen_dir / type_name).mkdir(exist_ok=True)
        
        # Group images by Pokemon
        pokemon_groups = group_images_by_pokemon(type_folder)
        
        print(f"   Found {len(pokemon_groups)} unique Pokemon")
        
        # Determine which Pokemon to hold out
        if holdout_pokemon_list:
            # Use specified Pokemon
            holdout_pokemon = set(
                name.lower().strip() 
                for name in holdout_pokemon_list 
                if name.lower().strip() in pokemon_groups
            )
        else:
            # Random selection
            holdout_pokemon = select_holdout_pokemon(
                pokemon_groups, holdout_ratio, min_images, seed
            )
        
        print(f"   Holding out {len(holdout_pokemon)} Pokemon for testing:")
        for name in sorted(holdout_pokemon):
            print(f"      - {name.capitalize()} ({len(pokemon_groups[name])} images)")
        
        # Split images
        type_stats = {
            'total_pokemon': len(pokemon_groups),
            'holdout_pokemon': len(holdout_pokemon),
            'train_images': 0,
            'test_unseen_images': 0,
            'test_seen_images': 0
        }
        
        for pokemon_name, images in pokemon_groups.items():
            if pokemon_name in holdout_pokemon:
                # All images of this Pokemon go to test_unseen
                for img in images:
                    shutil.copy2(img, test_unseen_dir / type_name / img.name)
                    type_stats['test_unseen_images'] += 1
                
                stats['holdout_pokemon_names'].append(f"{pokemon_name} ({type_name})")
            
            else:
                # Split this Pokemon's images: 80% train, 20% test_seen
                random.shuffle(images)
                split_point = int(len(images) * 0.8)
                
                train_images = images[:split_point]
                test_images = images[split_point:]
                
                for img in train_images:
                    shutil.copy2(img, train_dir / type_name / img.name)
                    type_stats['train_images'] += 1
                
                for img in test_images:
                    shutil.copy2(img, test_seen_dir / type_name / img.name)
                    type_stats['test_seen_images'] += 1
        
        stats['types'][type_name] = type_stats
        stats['total_pokemon'] += type_stats['total_pokemon']
        stats['holdout_pokemon'] += type_stats['holdout_pokemon']
        stats['train_images'] += type_stats['train_images']
        stats['test_unseen_images'] += type_stats['test_unseen_images']
        stats['test_seen_images'] += type_stats['test_seen_images']
    
    return stats


def main():
    args = parse_args()
    
    print("="*80)
    print("UNSEEN POKEMON TESTING - DATASET SPLIT")
    print("="*80)
    print(f"Source: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Holdout ratio: {args.holdout_ratio*100:.0f}%")
    if args.holdout_pokemon:
        print(f"Specific Pokemon: {args.holdout_pokemon}")
    print()
    
    # Parse holdout Pokemon list
    holdout_pokemon_list = None
    if args.holdout_pokemon:
        holdout_pokemon_list = [
            name.strip() for name in args.holdout_pokemon.split(',')
        ]
    
    # Create split
    print("Creating unseen Pokemon split...")
    print("-"*80)
    
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
    
    # Print summary
    print("\n" + "="*80)
    print("SPLIT COMPLETE")
    print("="*80)
    print(f"Total Pokemon: {stats['total_pokemon']}")
    print(f"Held-out Pokemon: {stats['holdout_pokemon']}")
    print()
    print(f"Training set:        {stats['train_images']:5} images (seen Pokemon)")
    print(f"Test (seen):         {stats['test_seen_images']:5} images (seen Pokemon)")
    print(f"Test (UNSEEN):       {stats['test_unseen_images']:5} images (NEVER SEEN!)")
    print()
    
    print("Held-out Pokemon (will test generalization):")
    for name in sorted(stats['holdout_pokemon_names'])[:20]:
        print(f"  - {name}")
    if len(stats['holdout_pokemon_names']) > 20:
        print(f"  ... and {len(stats['holdout_pokemon_names']) - 20} more")
    
    print()
    print(f"Stats saved to: {stats_file}")
    print()
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Train on the split dataset:")
    print(f"   python train_improved.py --data-dir {args.output_dir}/train")
    print()
    print("2. Test on SEEN Pokemon:")
    print(f"   python test_model.py --test-dir {args.output_dir}/test_seen")
    print()
    print("3. Test on UNSEEN Pokemon (the important one!):")
    print(f"   python test_model.py --test-dir {args.output_dir}/test_unseen")
    print()
    print("If 'test_unseen' performance is good → model LEARNED type features! ✓")
    print("If 'test_unseen' performance is poor → model MEMORIZED Pokemon ✗")
    print("="*80)


if __name__ == "__main__":
    main()