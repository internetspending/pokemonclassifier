"""
Train EfficientNet-B0 on Pokemon type classification with ANTI-OVERFITTING strategies.

Improvements over baseline:
  - Strong data augmentation (RandomResizedCrop, ColorJitter, RandomRotation, etc.)
  - Class-balanced sampling to handle imbalanced dataset
  - Label smoothing for regularization
  - Dropout in classifier head
  - Weight decay (L2 regularization)
  - Learning rate scheduling with warm restarts
  - Early stopping to prevent overfitting
  - Mixup augmentation (optional)

Usage:
    python classifiers/pytorch/train_improved.py
    python classifiers/pytorch/train_improved.py --augmentation strong --dropout 0.5
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from collections import Counter

from utils.dataset import PokemonDataset

# Import the experiment tracker
try:
    from experiment_tracker import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError:
    print("[WARNING] experiment_tracker.py not found. Running without experiment tracking.")
    TRACKER_AVAILABLE = False


def parse_args():
    p = argparse.ArgumentParser(description="Train Pokemon type classifier with anti-overfitting")
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "models"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (L2 regularization)")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in classifier head")
    p.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    p.add_argument("--seed", type=int, default=42)
    
    # Augmentation options
    p.add_argument("--augmentation", choices=["none", "light", "strong"], default="strong",
                   help="Data augmentation strength")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Mixup alpha (0 to disable, 0.2-0.4 recommended)")
    
    # Training strategy
    p.add_argument("--freeze-epochs", type=int, default=5,
                   help="Number of epochs to train with frozen backbone")
    p.add_argument("--class-balanced", action="store_true",
                   help="Use class-balanced sampling")
    p.add_argument("--early-stopping-patience", type=int, default=15,
                   help="Early stopping patience (0 to disable)")
    p.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine",
                   help="Learning rate scheduler")
    
    # Experiment tracking arguments
    p.add_argument("--experiment-name", default="pokemon_improved",
                   help="Name for this experiment")
    p.add_argument("--description", default=None,
                   help="Description of what you're testing")
    p.add_argument("--no-tracking", action="store_true",
                   help="Disable experiment tracking")
    
    return p.parse_args()


def get_augmentation_transforms(augmentation_type="strong", is_training=True):
    """Get data augmentation transforms based on augmentation strength."""
    
    # Common normalization (ImageNet stats)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Lambda to ensure RGB (handles RGBA images)
    ensure_rgb = transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img)
    
    if not is_training:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            ensure_rgb,  # Convert to RGB first
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    # Training transforms with augmentation
    if augmentation_type == "none":
        return transforms.Compose([
            ensure_rgb,  # Convert to RGB first
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == "light":
        return transforms.Compose([
            ensure_rgb,  # Convert to RGB first
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    
    elif augmentation_type == "strong":
        return transforms.Compose([
            ensure_rgb,  # Convert to RGB first
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            normalize
        ])


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

    return train_idx, val_idx, test_idx


def get_class_weights(dataset, indices):
    """Calculate class weights for balanced sampling."""
    labels = [dataset.samples[i][1] for i in indices]
    class_counts = Counter(labels)
    
    # Calculate weights (inverse frequency)
    total = len(labels)
    num_classes = len(class_counts)
    class_weights = {cls: total / (num_classes * count) for cls, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]
    
    return sample_weights, class_counts


class ImprovedClassifier(nn.Module):
    """EfficientNet-B0 with improved classifier head."""
    
    def __init__(self, num_classes=18, dropout=0.5):
        super().__init__()
        
        # Load pretrained backbone
        weights = EfficientNet_B0_Weights.DEFAULT
        efficientnet = efficientnet_b0(weights=weights)
        
        # Extract feature extractor
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Get feature dimension
        in_features = efficientnet.classifier[1].in_features
        
        # Improved classifier head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def freeze_backbone(model):
    """Freeze all backbone (feature extraction) layers."""
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all backbone layers."""
    for param in model.features.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """Run one training epoch with optional mixup."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if use_mixup and mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            # For mixup, use the dominant label for metrics
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels_a.cpu().tolist())
        else:
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


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def save_checkpoint(model, label_names, path, extra=None):
    """Save model checkpoint."""
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

    # Initialize experiment tracker
    tracker = None
    if TRACKER_AVAILABLE and not args.no_tracking:
        tracker = ExperimentTracker(
            experiment_name=args.experiment_name,
            description=args.description
        )
        print(f"[TRACKING] Experiment: {args.experiment_name}")
        print(f"[TRACKING] Results will be saved to: {tracker.get_experiment_path()}")
    
    # Load dataset with appropriate transforms
    print("\n=== Loading Dataset ===")
    base_dataset = PokemonDataset(data_dir=args.data_dir)
    train_idx, val_idx, test_idx = split_dataset(base_dataset, seed=args.seed)
    
    # Calculate class distribution
    sample_weights, class_counts = get_class_weights(base_dataset, train_idx)
    print(f"\nClass distribution in training set:")
    for cls, count in sorted(class_counts.items()):
        print(f"  Class {cls}: {count} samples")
    
    print(f"\nSplit: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
    print(f"Classes ({len(base_dataset.label_names)}): {base_dataset.label_names}")
    
    # Create datasets with transforms
    train_transform = get_augmentation_transforms(args.augmentation, is_training=True)
    val_transform = get_augmentation_transforms(args.augmentation, is_training=False)
    
    # Apply transforms to datasets
    train_dataset = PokemonDataset(data_dir=args.data_dir, transform=train_transform)
    val_dataset = PokemonDataset(data_dir=args.data_dir, transform=val_transform)
    test_dataset = PokemonDataset(data_dir=args.data_dir, transform=val_transform)
    
    train_set = Subset(train_dataset, train_idx)
    val_set = Subset(val_dataset, val_idx)
    test_set = Subset(test_dataset, test_idx)
    
    # Create data loaders with optional class balancing
    if args.class_balanced:
        print("\n[INFO] Using class-balanced sampling")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Log configuration
    config = {
        'model_architecture': 'efficientnet_b0',
        'dropout': args.dropout,
        'label_smoothing': args.label_smoothing,
        'weight_decay': args.weight_decay,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'freeze_epochs': args.freeze_epochs,
        'augmentation': args.augmentation,
        'mixup_alpha': args.mixup_alpha,
        'class_balanced': args.class_balanced,
        'scheduler': args.scheduler,
        'early_stopping_patience': args.early_stopping_patience,
        'seed': args.seed,
        'num_classes': len(base_dataset.label_names),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
    }
    
    if tracker:
        tracker.log_config(config)
    
    # Build model with dropout
    print(f"\n=== Building Model ===")
    print(f"Dropout: {args.dropout}")
    print(f"Label Smoothing: {args.label_smoothing}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Augmentation: {args.augmentation}")
    
    model = ImprovedClassifier(
        num_classes=len(base_dataset.label_names),
        dropout=args.dropout
    ).to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print(f"Using Cosine Annealing scheduler")
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
        print(f"Using ReduceLROnPlateau scheduler")
    
    # Early stopping
    early_stopping = None
    if args.early_stopping_patience > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
        print(f"Using early stopping with patience={args.early_stopping_patience}")
    
    best_val_f1 = 0.0
    ckpt_path = os.path.join(args.output_dir, "best_pokemon_classifier_improved.pt")
    
    # Phase 1: Train with frozen backbone
    if args.freeze_epochs > 0:
        print(f"\n=== Phase 1: Training with frozen backbone ({args.freeze_epochs} epochs) ===")
        freeze_backbone(model)
    
        for epoch in range(1, args.freeze_epochs + 1):
            train_loss, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=(args.mixup_alpha > 0), mixup_alpha=args.mixup_alpha
            )
            val_loss, val_f1, val_top5_acc = validate(model, val_loader, criterion, device)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"[Epoch {epoch}/{args.freeze_epochs}] "
                  f"LR={current_lr:.2e} | "
                  f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
                  f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} top5={val_top5_acc:.4f}")
            
            # Log to tracker
            if tracker:
                tracker.log_epoch_metrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_f1=train_f1,
                    val_loss=val_loss,
                    val_f1=val_f1,
                    val_top5_acc=val_top5_acc,
                    learning_rate=current_lr
                )
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_checkpoint(model, base_dataset.label_names, ckpt_path,
                              extra={"epoch": epoch, "val_f1": val_f1})
                print(f"  ✓ New best model! (val_f1={val_f1:.4f})")
                
                if tracker:
                    tracker.save_checkpoint(model, optimizer, epoch, val_f1, is_best=True)
            
            # Update scheduler
            if scheduler and args.scheduler == "cosine":
                scheduler.step()
    
    # Phase 2: Fine-tune entire model
    print(f"\n=== Phase 2: Fine-tuning entire model ({args.epochs - args.freeze_epochs} epochs) ===")
    unfreeze_backbone(model)
    
    for epoch in range(args.freeze_epochs + 1, args.epochs + 1):
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(args.mixup_alpha > 0), mixup_alpha=args.mixup_alpha
        )
        val_loss, val_f1, val_top5_acc = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch}/{args.epochs}] "
              f"LR={current_lr:.2e} | "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} top5={val_top5_acc:.4f}")
        
        # Log to tracker
        if tracker:
            tracker.log_epoch_metrics(
                epoch=epoch,
                train_loss=train_loss,
                train_f1=train_f1,
                val_loss=val_loss,
                val_f1=val_f1,
                val_top5_acc=val_top5_acc,
                learning_rate=current_lr
            )
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(model, base_dataset.label_names, ckpt_path,
                          extra={"epoch": epoch, "val_f1": val_f1})
            print(f"  ✓ New best model! (val_f1={val_f1:.4f})")
            
            if tracker:
                tracker.save_checkpoint(model, optimizer, epoch, val_f1, is_best=True)
        
        # Update scheduler
        if scheduler:
            if args.scheduler == "cosine":
                scheduler.step()
            elif args.scheduler == "plateau":
                scheduler.step(val_f1)
        
        # Early stopping check
        if early_stopping:
            if early_stopping(val_f1):
                print(f"\n[EARLY STOPPING] No improvement for {args.early_stopping_patience} epochs")
                break
    
    # Test evaluation
    print("\n=== Test Set Evaluation (best checkpoint) ===")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_loss, test_f1, test_top5_acc = validate(model, test_loader, criterion, device)
    
    print(f"\nFinal Results:")
    print(f"  Best Val F1:  {best_val_f1:.4f}")
    print(f"  Test F1:      {test_f1:.4f}")
    print(f"  Test Top-5:   {test_top5_acc:.4f}")
    print(f"\nCheckpoint saved to: {ckpt_path}")
    
    # Save final summary
    if tracker:
        tracker.metrics['test_f1'] = [test_f1]
        tracker.metrics['test_top5_acc'] = [test_top5_acc]
        tracker.save_final_summary()
        
        print(f"\n[TRACKING] All results saved to: {tracker.get_experiment_path()}")


if __name__ == "__main__":
    main()
