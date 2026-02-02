# Pokemon Type Classifier

**Authors:** Saigaurav Bettela, Jeron Perey, Nick Perlich, Sean McCormick

## Motivation
We aim to classify a Pokémon’s **primary type** (18 classes) from an image using transfer learning with a pretrained CNN.

## Problem
Given a Pokémon image, predict its primary type (e.g., Fire, Water, Electric) using learned visual features.

## Approach
- Supervised learning using a Kaggle dataset (images + type labels)
- Transfer learning (e.g., ResNet / EfficientNet)
- Train head first (freeze backbone), then fine-tune upper layers

## Evaluation
- Macro F1 (primary metric)
- Top-k accuracy (top-1 and top-3/top-5)
- Train/val/test split: 70/10/20

## Repo Structure
- `src/` training/eval code
- `notebooks/` experiments/EDA
- `data/` dataset (not tracked)
- `models/` saved checkpoints (not tracked)
- `reports/` figures + results

## Setup
1. Create env and install deps
2. Download dataset into `data/`
3. Train
4. Evaluate

## Commands (example)
- Train: `python -m src.training.train`
- Eval:  `python -m src.evaluation.eval`
