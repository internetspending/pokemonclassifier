"""
Automated Hyperparameter Search for Pokemon Classifier

This script automatically runs multiple training experiments with different
hyperparameter combinations and tracks all results.

Usage:
    python hyperparameter_search.py --search-type grid
    python hyperparameter_search.py --search-type random --num-trials 10
    python hyperparameter_search.py --config search_config.yaml
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Automated hyperparameter search")
    p.add_argument("--search-type", choices=["grid", "random"], default="grid",
                   help="Type of search: 'grid' for grid search, 'random' for random search")
    p.add_argument("--num-trials", type=int, default=None,
                   help="Number of random trials (only for random search)")
    p.add_argument("--config", default=None,
                   help="Path to YAML config file with search space")
    p.add_argument("--train-script", default="classifiers/pytorch/train.py",
                   help="Path to training script")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without running them")
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of experiments to run in parallel (NOT IMPLEMENTED YET)")
    return p.parse_args()


# ============================================
# DEFINE YOUR SEARCH SPACE HERE
# ============================================

SEARCH_SPACE_GRID = {
    "phase1_lr": [1e-3, 5e-4, 1e-4],
    "phase2_lr": [1e-4, 5e-5, 1e-5],
    "batch_size": [16, 32],
    "phase1_epochs": [10],
    "phase2_epochs": [10],
}

SEARCH_SPACE_RANDOM = {
    "phase1_lr": (1e-4, 1e-3),      # (min, max) for continuous
    "phase2_lr": (1e-5, 1e-4),       # (min, max)
    "batch_size": [16, 32, 64],      # list for discrete choices
    "phase1_epochs": [10, 15, 20],
    "phase2_epochs": [10, 15, 20],
}

# Fixed hyperparameters (used in all experiments)
FIXED_PARAMS = {
    "seed": 42,
}


def load_config_file(config_path):
    """Load search space from YAML config file"""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('search_space', {}), config.get('fixed_params', {})


def generate_grid_search_configs(search_space):
    """Generate all combinations for grid search"""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)
    
    return configs


def sample_random_config(search_space):
    """Sample one random configuration"""
    config = {}
    
    for param, value in search_space.items():
        if isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value, list):
            # Continuous range (min, max)
            min_val, max_val = value
            # Log-uniform sampling for learning rates
            if 'lr' in param.lower():
                config[param] = 10 ** random.uniform(
                    math.log10(min_val), 
                    math.log10(max_val)
                )
            else:
                config[param] = random.uniform(min_val, max_val)
        elif isinstance(value, list):
            # Discrete choices
            config[param] = random.choice(value)
        elif isinstance(value, tuple):
            # Continuous range specified as tuple
            min_val, max_val = value
            if 'lr' in param.lower():
                config[param] = 10 ** random.uniform(
                    math.log10(min_val), 
                    math.log10(max_val)
                )
            else:
                config[param] = random.uniform(min_val, max_val)
    
    return config


def generate_random_search_configs(search_space, num_trials):
    """Generate random configurations"""
    configs = []
    for _ in range(num_trials):
        config = sample_random_config(search_space)
        configs.append(config)
    return configs


def config_to_experiment_name(config):
    """Generate a descriptive experiment name from config"""
    parts = []
    
    # Add key parameters to name
    if 'phase1_lr' in config:
        parts.append(f"lr1_{config['phase1_lr']:.0e}")
    if 'phase2_lr' in config:
        parts.append(f"lr2_{config['phase2_lr']:.0e}")
    if 'batch_size' in config:
        parts.append(f"bs{config['batch_size']}")
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%m%d_%H%M")
    parts.append(timestamp)
    
    return "autosearch_" + "_".join(parts)


def config_to_description(config):
    """Generate a description from config"""
    parts = []
    for key, value in config.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.0e}")
        else:
            parts.append(f"{key}={value}")
    return "Auto search: " + ", ".join(parts)


def build_command(train_script, config, fixed_params):
    """Build the command to run training with given config"""
    cmd = ["python", train_script]
    
    # Merge config with fixed params
    all_params = {**config, **fixed_params}
    
    # Add experiment name and description
    exp_name = config_to_experiment_name(config)
    exp_desc = config_to_description(config)
    
    cmd.extend(["--experiment-name", exp_name])
    cmd.extend(["--description", exp_desc])
    
    # Add all hyperparameters
    for key, value in all_params.items():
        # Convert python parameter names to command-line flags
        flag = f"--{key.replace('_', '-')}"
        cmd.extend([flag, str(value)])
    
    return cmd, exp_name


def run_experiment(cmd, exp_name, dry_run=False):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"Running Experiment: {exp_name}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print("="*80)
    
    if dry_run:
        print("[DRY RUN] Skipping actual execution")
        return {"status": "dry_run", "experiment_name": exp_name}
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ Experiment '{exp_name}' completed successfully")
        return {"status": "success", "experiment_name": exp_name}
    
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{exp_name}' failed with error code {e.returncode}")
        return {"status": "failed", "experiment_name": exp_name, "error": str(e)}
    
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment '{exp_name}' interrupted by user")
        return {"status": "interrupted", "experiment_name": exp_name}


def save_search_summary(results, search_log_path):
    """Save summary of all experiments in the search"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "experiments": results
    }
    
    with open(search_log_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSearch summary saved to: {search_log_path}")


def print_summary(results):
    """Print a summary of the hyperparameter search"""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)
    
    total = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print("\nExperiments run:")
    for i, result in enumerate(results, 1):
        status_symbol = "✓" if result["status"] == "success" else "✗"
        print(f"  {i}. {status_symbol} {result['experiment_name']} - {result['status']}")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Run: python view_experiments.py")
    print("  2. Compare all experiments to find the best one")
    print("  3. Look at the 'experiments/' directory for detailed results")
    print("="*80 + "\n")


def main():
    args = parse_args()
    
    # Check if train script exists
    if not os.path.exists(args.train_script):
        print(f"Error: Training script not found at {args.train_script}")
        sys.exit(1)
    
    # Load search space
    if args.config:
        search_space, fixed_params = load_config_file(args.config)
    else:
        if args.search_type == "grid":
            search_space = SEARCH_SPACE_GRID
        else:
            search_space = SEARCH_SPACE_RANDOM
        fixed_params = FIXED_PARAMS
    
    # Generate configurations
    print("="*80)
    print("HYPERPARAMETER SEARCH CONFIGURATION")
    print("="*80)
    print(f"Search type: {args.search_type}")
    
    if args.search_type == "grid":
        configs = generate_grid_search_configs(search_space)
        print(f"Total combinations: {len(configs)}")
    else:
        num_trials = args.num_trials or 10
        configs = generate_random_search_configs(search_space, num_trials)
        print(f"Number of random trials: {len(configs)}")
    
    print("\nSearch space:")
    for key, value in search_space.items():
        print(f"  {key}: {value}")
    
    print("\nFixed parameters:")
    for key, value in fixed_params.items():
        print(f"  {key}: {value}")
    
    # Ask for confirmation
    if not args.dry_run:
        response = input(f"\nThis will run {len(configs)} experiments. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Run all experiments
    print("\n" + "="*80)
    print(f"STARTING {len(configs)} EXPERIMENTS")
    print("="*80)
    
    results = []
    
    try:
        for i, config in enumerate(configs, 1):
            print(f"\n\n{'='*80}")
            print(f"EXPERIMENT {i}/{len(configs)}")
            print(f"{'='*80}")
            
            cmd, exp_name = build_command(args.train_script, config, fixed_params)
            result = run_experiment(cmd, exp_name, dry_run=args.dry_run)
            results.append(result)
            
            # Save intermediate results
            search_log_path = "hyperparameter_search_results.json"
            save_search_summary(results, search_log_path)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Search interrupted by user")
    
    # Print final summary
    print_summary(results)
    
    # Save final results
    search_log_path = "hyperparameter_search_results.json"
    save_search_summary(results, search_log_path)


if __name__ == "__main__":
    import math  # For log calculations
    main()