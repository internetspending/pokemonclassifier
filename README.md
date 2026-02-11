# Pokemon Type Classifier

**Authors:** Saigaurav Bettela, Jeron Perey, Nick Perlich, Sean McCormick

## Motivation
We aim to classify a Pokémon’s **primary type** (18 classes) from an image using transfer learning with a pretrained CNN.

## Dataset access link
https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types?select=pokemon.csv

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
- `data/` dataset (not tracked)
- `classifiers/` model implementations
  - `pytorch/` PyTorch-based classifiers
  - `tensorflow/` TensorFlow-based classifiers
  - `baseline/` baseline models + EDA
- `evaluations/` evaluation code + results
- `utils/` shared utilities (dataset, preprocessing)
- `scripts/` helper scripts
- `requirements/` dependency files (base, cpu, cu126)

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   # For CPU-only:
   pip install -r requirements/requirements.cpu.txt
   # For CUDA 12.6:
   pip install -r requirements/requirements.cu126.txt
   ```
2. Download the dataset from the [Kaggle link above](https://www.kaggle.com/datasets/vishalsubbiah/pokemon-images-and-types?select=pokemon.csv) and place the files so the structure looks like:
   ```
   data/
   ├── pokemon.csv
   └── images/
       ├── bulbasaur.png
       ├── charmander.png
       └── ...
   ```
3. Train
4. Evaluate
