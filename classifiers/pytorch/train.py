"""
Train EfficientNet-B0 on the Pokemon type classification task with experiment tracking.

Two-phase transfer learning:
  Phase 1 — Freeze backbone, train only the classifier head.
  Phase 2 — Unfreeze upper backbone layers and fine-tune.

Usage:
    python classifiers/pytorch/train.py
    python classifiers/pytorch/train.py --phase1-epochs 5 --phase2-epochs 5
    python classifiers/pytorch/train.py --experiment-name "efficientnet_b0_baseline" --description "Testing baseline model"
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils.dataset import PokemonDataset

# Import the experiment tracker
try:
    from utils.experiment_tracker import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError:
    print("[WARNING] experiment_tracker.py not found. Running without experiment tracking.")
    TRACKER_AVAILABLE = False


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
    
    # Experiment tracking arguments
    p.add_argument("--experiment-name", default="pokemon_classifier_training",
                   help="Name for this experiment (used in tracking)")
    p.add_argument("--description", default=None,
                   help="Description of what you're testing in this experiment")
    p.add_argument("--no-tracking", action="store_true",
                   help="Disable experiment tracking")
    
    return p.parse_args()


def split_dataset(dataset, seed=42):
    """Stratified 70/10/20 train/val/test split."""
    labels = [label for _, label in dataset.samples]
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
    """Load pretrained EfficientNet-B0 and replace the classifier head."""
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model):
    """Freeze all backbone (feature extraction) layers."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_upper_layers(model, num_blocks_to_unfreeze=3):
    """Unfreeze the last N blocks of the backbone."""
    total_blocks = len(model.features)
    for i in range(total_blocks - num_blocks_to_unfreeze, total_blocks):
        for param in model.features[i].parameters():
            param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns (avg_loss, macro_f1)."""
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
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


@torch.inference_mode()
def validate(model, loader, criterion, device):
    """Run validation. Returns (avg_loss, macro_f1, top5_acc)."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    top5_correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        # Calculate top-5 accuracy
        _, top5_pred = logits.topk(5, 1, True, True)
        top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    top5_acc = top5_correct / total
    
    return avg_loss, macro_f1, top5_acc


@torch.inference_mode()
def topk_accuracy(model, loader, device, ks=(1, 3, 5)):
    """Compute top-k accuracy for the given k values."""
    model.eval()
    correct = {k: 0 for k in ks}
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        for k in ks:
            top_k_preds = logits.topk(k, dim=1).indices
            correct[k] += (top_k_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    return {k: correct[k] / total for k in ks}


def evaluate_test(model, test_loader, device, label_names):
    """Full test set evaluation: macro F1 + top-k accuracy."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="  test", leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    topk = topk_accuracy(model, test_loader, device)

    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Top-1 Accuracy: {topk[1]:.4f}")
    print(f"  Top-3 Accuracy: {topk[3]:.4f}")
    print(f"  Top-5 Accuracy: {topk[5]:.4f}")
    
    return macro_f1, topk


def save_checkpoint(model, label_names, path, extra=None):
    """Save model checkpoint with label metadata."""
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

    # ========================================
    # INITIALIZE EXPERIMENT TRACKER
    # ========================================
    tracker = None
    if TRACKER_AVAILABLE and not args.no_tracking:
        tracker = ExperimentTracker(
            experiment_name=args.experiment_name,
            description=args.description
        )
        print(f"[TRACKING] Experiment: {args.experiment_name}")
        print(f"[TRACKING] Results will be saved to: {tracker.get_experiment_path()}")
    
    # Load dataset and split
    dataset = PokemonDataset(data_dir=args.data_dir)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    print(f"Split: {len(train_set)} train / {len(val_set)} val / {len(test_set)} test")
    print(f"Classes ({len(dataset.label_names)}): {dataset.label_names}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # ========================================
    # LOG CONFIGURATION TO TRACKER
    # ========================================
    config = {
        # Model settings
        'model_architecture': 'efficientnet_b0',
        'pretrained': True,
        'num_classes': len(dataset.label_names),
        
        # Training settings
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
        'total_epochs': args.phase1_epochs + args.phase2_epochs,
        'batch_size': args.batch_size,
        'phase1_lr': args.phase1_lr,
        'phase2_lr': args.phase2_lr,
        
        # Data settings
        'train_size': len(train_set),
        'val_size': len(val_set),
        'test_size': len(test_set),
        'train_split': 0.7,
        'val_split': 0.1,
        'test_split': 0.2,
        
        # Other
        'seed': args.seed,
        'device': device,
    }
    
    if tracker:
        tracker.log_config(config)
    
    # Build model
    model = build_model(num_classes=len(dataset.label_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_pokemon_classifier.pt")

    # ── Phase 1: Frozen backbone ──
    print("\n=== Phase 1: Train classifier head (backbone frozen) ===")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr,
    )
    
    global_epoch = 0  # Track overall epoch number for logging

    for epoch in range(1, args.phase1_epochs + 1):
        global_epoch += 1
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_top5_acc = validate(model, val_loader, criterion, device)

        print(f"[Phase1 {epoch}/{args.phase1_epochs}] "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
              f"val_top5_acc={val_top5_acc:.4f}")

        # ========================================
        # LOG METRICS TO TRACKER
        # ========================================
        if tracker:
            tracker.log_epoch_metrics(
                epoch=global_epoch,
                train_loss=train_loss,
                train_f1=train_f1,
                val_loss=val_loss,
                val_f1=val_f1,
                val_top5_acc=val_top5_acc,
                learning_rate=args.phase1_lr
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, dataset.label_names, ckpt_path,
                            extra={"phase": 1, "epoch": epoch, "val_f1": val_f1})
            print(f"  -> Saved best checkpoint (val_f1={val_f1:.4f})")
            
            # Save checkpoint to tracker too
            if tracker:
                tracker.save_checkpoint(model, optimizer, global_epoch, val_f1, is_best=True)

    # ── Phase 2: Fine-tune upper layers ──
    print("\n=== Phase 2: Fine-tune upper layers ===")
    unfreeze_upper_layers(model, num_blocks_to_unfreeze=3)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr,
    )

    for epoch in range(1, args.phase2_epochs + 1):
        global_epoch += 1
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_top5_acc = validate(model, val_loader, criterion, device)

        print(f"[Phase2 {epoch}/{args.phase2_epochs}] "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | "
              f"val_top5_acc={val_top5_acc:.4f}")

        # LOG METRICS TO TRACKER
        if tracker:
            tracker.log_epoch_metrics(
                epoch=global_epoch,
                train_loss=train_loss,
                train_f1=train_f1,
                val_loss=val_loss,
                val_f1=val_f1,
                val_top5_acc=val_top5_acc,
                learning_rate=args.phase2_lr
            )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, dataset.label_names, ckpt_path,
                            extra={"phase": 2, "epoch": epoch, "val_f1": val_f1})
            print(f"  -> Saved best checkpoint (val_f1={val_f1:.4f})")
            
            # Save checkpoint to tracker too
            if tracker:
                tracker.save_checkpoint(model, optimizer, global_epoch, val_f1, is_best=True)

    # ── Test Evaluation ──
    print("\n=== Test Set Evaluation (best checkpoint) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_f1, test_topk = evaluate_test(model, test_loader, device, dataset.label_names)

    print(f"\nBest val F1: {best_f1:.4f}")
    print(f"Checkpoint saved to: {ckpt_path}")

    # SAVE FINAL SUMMARY
    # Run: python train.py --experiment-name "my_first_tracked_experiment"
    if tracker:
        # Log test metrics to the final summary manually
        tracker.metrics['test_f1'] = [test_f1]
        tracker.metrics['test_top1_acc'] = [test_topk[1]]
        tracker.metrics['test_top3_acc'] = [test_topk[3]]
        tracker.metrics['test_top5_acc'] = [test_topk[5]]
        
        tracker.save_final_summary()
        
        print(f"\n[TRACKING] All results saved to: {tracker.get_experiment_path()}")
        print(f"[TRACKING] View plots at: {tracker.get_experiment_path()}/plots/")


if __name__ == "__main__":
    main()