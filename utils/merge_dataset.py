"""
Merge Pokemon Generation One Dataset (Pokemon Name Folders → Type Folders)

This script merges the Gen 1 dataset organized by Pokemon names into your
existing dataset organized by Pokemon types.

Structure:
  data/dataset/Pikachu/*.png  →  data/images/Electric/pikachu_*.png
  data/dataset/Charizard/*.png → data/images/Fire/charizard_*.png

Usage:
    python merge_gen1_by_name.py
    python merge_gen1_by_name.py --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Merge Gen 1 Pokemon dataset by name")
    p.add_argument("--source", default="data/dataset",
                   help="Source directory with Pokemon name folders (default: data/dataset)")
    p.add_argument("--target", default="data/images",
                   help="Target directory with type folders (default: data/images)")
    p.add_argument("--csv", default="data/pokemon.csv",
                   help="CSV file with Pokemon types (default: data/pokemon.csv)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without doing it")
    p.add_argument("--skip-duplicates", action="store_true",
                   help="Skip Pokemon that already exist in target")
    return p.parse_args()


def load_pokemon_types(csv_path):
    """Load Pokemon name → primary type mapping from CSV"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Could not read CSV: {e}")
        return None
    
    # Find the right columns (handle different CSV formats)
    name_cols = ['Name', 'name', 'Pokemon', 'pokemon', '#']
    type_cols = ['Type1', 'type1', 'Type', 'type', 'Primary Type', 'primary_type']
    
    name_col = None
    type_col = None
    
    for col in name_cols:
        if col in df.columns:
            name_col = col
            break
    
    for col in type_cols:
        if col in df.columns:
            type_col = col
            break
    
    if not name_col or not type_col:
        print(f"⚠️  Could not find name/type columns")
        print(f"   Available columns: {df.columns.tolist()}")
        return None
    
    # Create mapping: lowercase name → type
    type_mapping = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).lower().strip()
        poke_type = str(row[type_col]).strip()
        type_mapping[name] = poke_type
    
    print(f"✓ Loaded types for {len(type_mapping)} Pokemon from CSV")
    return type_mapping


def normalize_name(name):
    """Normalize Pokemon name for matching"""
    # Convert to lowercase and remove special characters
    name = name.lower().strip()
    name = name.replace('-', '').replace('_', '').replace(' ', '')
    name = name.replace('.', '').replace("'", '')
    return name


def find_pokemon_type(folder_name, type_mapping):
    """Find the type for a Pokemon folder"""
    # Try direct match
    normalized = normalize_name(folder_name)
    
    if normalized in type_mapping:
        return type_mapping[normalized]
    
    # Try all mappings
    for pokemon_name, poke_type in type_mapping.items():
        if normalize_name(pokemon_name) == normalized:
            return poke_type
    
    # Try partial match (for variations like "Charizard Mega X")
    for pokemon_name, poke_type in type_mapping.items():
        if normalized.startswith(normalize_name(pokemon_name)):
            return poke_type
        if normalize_name(pokemon_name).startswith(normalized):
            return poke_type
    
    return None


def count_images(directory):
    """Count total images in directory"""
    total = 0
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        total += len(list(Path(directory).rglob(ext)))
    return total


def merge_datasets(source_dir, target_dir, type_mapping, skip_duplicates, dry_run):
    """Merge Pokemon name folders into type folders"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Valid Pokemon types
    valid_types = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 
                   'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 
                   'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
    
    stats = {
        'total_pokemon': 0,
        'total_images': 0,
        'copied': 0,
        'skipped_no_type': 0,
        'skipped_invalid_type': 0,
        'skipped_duplicate': 0,
        'by_type': defaultdict(int),
        'unmapped_pokemon': []
    }
    
    # Get all Pokemon folders
    pokemon_folders = [d for d in source_dir.iterdir() if d.is_dir()]
    stats['total_pokemon'] = len(pokemon_folders)
    
    print(f"\n📁 Found {len(pokemon_folders)} Pokemon folders in source")
    print("-" * 80)
    
    # Process each Pokemon folder
    for pokemon_folder in sorted(pokemon_folders):
        pokemon_name = pokemon_folder.name
        
        # Find type for this Pokemon
        poke_type = find_pokemon_type(pokemon_name, type_mapping)
        
        if not poke_type:
            stats['skipped_no_type'] += 1
            stats['unmapped_pokemon'].append(pokemon_name)
            print(f"⚠️  {pokemon_name}: No type found")
            continue
        
        if poke_type not in valid_types:
            stats['skipped_invalid_type'] += 1
            print(f"⚠️  {pokemon_name}: Invalid type '{poke_type}'")
            continue
        
        # Get all images in this Pokemon's folder
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(list(pokemon_folder.glob(ext)))
        
        if not image_files:
            print(f"⚠️  {pokemon_name}: No images found")
            continue
        
        stats['total_images'] += len(image_files)
        
        # Create target type folder
        type_folder = target_dir / poke_type
        if not dry_run:
            type_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy each image
        copied_count = 0
        for img_file in image_files:
            # Create new filename: pokemonname_original.png
            base_name = normalize_name(pokemon_name)
            new_name = f"{base_name}_{img_file.stem}{img_file.suffix}"
            target_path = type_folder / new_name
            
            # Handle duplicates
            if target_path.exists():
                if skip_duplicates:
                    stats['skipped_duplicate'] += 1
                    continue
                else:
                    counter = 1
                    while target_path.exists():
                        new_name = f"{base_name}_{img_file.stem}_{counter}{img_file.suffix}"
                        target_path = type_folder / new_name
                        counter += 1
            
            # Copy image
            if dry_run:
                if copied_count == 0:  # Only print first image per Pokemon
                    print(f"✓ {pokemon_name} ({poke_type}): {len(image_files)} images")
            else:
                try:
                    shutil.copy2(img_file, target_path)
                    copied_count += 1
                except Exception as e:
                    print(f"❌ Error copying {img_file}: {e}")
                    continue
            
            stats['copied'] += 1
            stats['by_type'][poke_type] += 1
        
        if not dry_run and copied_count > 0:
            print(f"✓ {pokemon_name} ({poke_type}): {copied_count} images copied")
    
    return stats


def main():
    args = parse_args()
    
    print("=" * 80)
    print("POKEMON GENERATION ONE DATASET MERGER")
    print("=" * 80)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"CSV: {args.csv}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()
    
    # Check paths exist
    source_path = Path(args.source)
    target_path = Path(args.target)
    csv_path = Path(args.csv)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {args.source}")
        return
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {args.csv}")
        print("   Please make sure pokemon.csv is in the data/ directory")
        return
    
    # Load Pokemon types
    print("📖 Loading Pokemon types from CSV...")
    type_mapping = load_pokemon_types(csv_path)
    
    if not type_mapping:
        print("❌ Could not load Pokemon types")
        return
    
    # Count current dataset
    current_count = count_images(target_path) if target_path.exists() else 0
    source_count = count_images(source_path)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Current dataset: {current_count} images")
    print(f"   Source dataset: {source_count} images")
    
    # Confirm
    if not args.dry_run:
        print()
        response = input("Proceed with merge? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    print("\n" + "=" * 80)
    print("MERGING DATASETS...")
    print("=" * 80)
    
    # Merge
    stats = merge_datasets(source_path, target_path, type_mapping, 
                          args.skip_duplicates, args.dry_run)
    
    # Print summary
    print("\n" + "=" * 80)
    print("MERGE COMPLETE")
    print("=" * 80)
    print(f"Pokemon folders processed: {stats['total_pokemon']}")
    print(f"Total images found: {stats['total_images']}")
    print(f"Images copied: {stats['copied']}")
    print(f"Skipped (no type found): {stats['skipped_no_type']}")
    print(f"Skipped (invalid type): {stats['skipped_invalid_type']}")
    print(f"Skipped (duplicate): {stats['skipped_duplicate']}")
    
    if stats['unmapped_pokemon']:
        print(f"\n⚠️  Unmapped Pokemon ({len(stats['unmapped_pokemon'])}):")
        for name in sorted(stats['unmapped_pokemon'])[:10]:  # Show first 10
            print(f"   - {name}")
        if len(stats['unmapped_pokemon']) > 10:
            print(f"   ... and {len(stats['unmapped_pokemon']) - 10} more")
    
    print("\n📊 Images added per type:")
    for poke_type in sorted(stats['by_type'].keys()):
        count = stats['by_type'][poke_type]
        print(f"   {poke_type:12} +{count:4} images")
    
    new_total = current_count + stats['copied']
    print()
    print(f"Dataset size: {current_count} → {new_total} (+{stats['copied']} images)")
    
    if not args.dry_run:
        print(f"\n✓ Dataset successfully merged to: {args.target}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()