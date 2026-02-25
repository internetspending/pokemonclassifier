"""
Train EfficientNet-B0 on the Pokemon type classification task.

Two-phase transfer learning:
  Phase 1 — Freeze backbone, train only the classifier head.
  Phase 2 — Unfreeze upper backbone layers and fine-tune.

Evaluation uses dual-type accuracy: a prediction is correct if it
matches either Type1 or Type2 of the Pokemon.

Usage:
    python -m scripts.train                          (from project root)
    python -m scripts.train --phase1-epochs 5 --phase2-epochs 10
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.dataset import PokemonDataset


def parse_args():
    p = argparse.ArgumentParser(description="Train Pokemon type classifier")
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "models"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--phase1-epochs", type=int, default=10)
    p.add_argument("--phase2-epochs", type=int, default=10)
    p.add_argument("--phase1-lr", type=float, default=1e-3)
    p.add_argument("--phase2-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_dataset(dataset, seed=42):
    """Stratified 70/10/20 train/val/test split on Type1."""
    labels = [type1_idx for _, type1_idx, _ in dataset.samples]
    indices = list(range(len(dataset)))

    # 80% train+val, 20% test
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )

    # From the 80%, take 12.5% for val -> 70/10/20 overall
    trainval_labels = [labels[i] for i in trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.125, random_state=seed, stratify=trainval_labels
    )

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_model(num_classes=18):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_upper_layers(model, num_blocks_to_unfreeze=3):
    total_blocks = len(model.features)
    for i in range(total_blocks - num_blocks_to_unfreeze, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """One training epoch. Returns (avg_loss, macro_f1 against type1)."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


@torch.inference_mode()
def dual_type_topk_accuracy(model, subset, full_dataset, device, k=1):
    """
    Top-k accuracy where a hit is counted if ANY of the top-k predictions
    matches type1 OR type2 for that Pokemon.
    """
    model.eval()
    correct = 0
    for orig_idx in subset.indices:
        img, _ = full_dataset[orig_idx]
        valid = full_dataset.valid_labels(orig_idx)
        logits = model(img.unsqueeze(0).to(device))
        topk_preds = logits.topk(k, dim=1).indices[0].tolist()
        if any(p in valid for p in topk_preds):
            correct += 1
    return correct / len(subset.indices)


def save_checkpoint(model, label_names, path, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "label_names": label_names,
        "num_classes": len(label_names),
    }
    if extra:
        state.update(extra)
    torch.save(state, path)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset & split
    dataset = PokemonDataset(data_dir=args.data_dir)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")
    print(f"Classes ({len(dataset.label_names)}): {dataset.label_names}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # Build model
    model = build_model(num_classes=len(dataset.label_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_pokemon_classifier.pt")

    # ── Phase 1: Frozen backbone ─────────────────────────────────────────────
    print("\n=== Phase 1: Train classifier head (backbone frozen) ===")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr,
    )

    for epoch in range(1, args.phase1_epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = dual_type_topk_accuracy(model, val_set, dataset, device, k=1)

        print(f"[Phase1 {epoch:>2}/{args.phase1_epochs}]  "
              f"loss={train_loss:.4f}  train_f1={train_f1:.4f}  "
              f"val_acc(dual-type)={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, dataset.label_names, ckpt_path,
                            extra={"phase": 1, "epoch": epoch, "val_acc": val_acc})
            print(f"             ^ new best — checkpoint saved")

    # ── Phase 2: Fine-tune upper layers ──────────────────────────────────────
    print("\n=== Phase 2: Fine-tune upper layers ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    unfreeze_upper_layers(model, num_blocks_to_unfreeze=3)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr,
    )

    for epoch in range(1, args.phase2_epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = dual_type_topk_accuracy(model, val_set, dataset, device, k=1)

        print(f"[Phase2 {epoch:>2}/{args.phase2_epochs}]  "
              f"loss={train_loss:.4f}  train_f1={train_f1:.4f}  "
              f"val_acc(dual-type)={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, dataset.label_names, ckpt_path,
                            extra={"phase": 2, "epoch": epoch, "val_acc": val_acc})
            print(f"             ^ new best — checkpoint saved")

    # ── Test Evaluation ───────────────────────────────────────────────────────
    print("\n=== Test Set Evaluation (best checkpoint) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    top1 = dual_type_topk_accuracy(model, test_set, dataset, device, k=1)
    top3 = dual_type_topk_accuracy(model, test_set, dataset, device, k=3)
    top5 = dual_type_topk_accuracy(model, test_set, dataset, device, k=5)

    # F1 against type1 (single ground-truth label required by sklearn)
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for orig_idx in test_set.indices:
            img, label = dataset[orig_idx]
            logits = model(img.unsqueeze(0).to(device))
            all_preds.append(logits.argmax(dim=1).item())
            all_labels.append(label)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"  Top-1 accuracy (dual-type): {top1:.4f}")
    print(f"  Top-3 accuracy (dual-type): {top3:.4f}")
    print(f"  Top-5 accuracy (dual-type): {top5:.4f}")
    print(f"  Macro F1 (vs type1):        {macro_f1:.4f}")
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
