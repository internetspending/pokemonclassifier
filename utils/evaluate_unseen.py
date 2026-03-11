"""
Test Model on Unseen Pokemon

Tests a trained model on Pokemon it has never seen during training.

Usage:
    python evaluate_unseen.py \
        --checkpoint models/best_pokemon_classifier_improved.pt \
        --test-dir data_unseen_pokemon/test_unseen
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import PokemonDataset


def parse_args():
    p = argparse.ArgumentParser(description="Test on unseen Pokemon")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint")
    p.add_argument("--test-dir", required=True,
                   help="Directory with test images (organized by type)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


class ImprovedClassifier(nn.Module):
    """Same model architecture as training"""
    def __init__(self, num_classes=18, dropout=0.5):
        super().__init__()
        weights = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'
        )
        from torchvision.models import efficientnet_b0
        efficientnet = efficientnet_b0()
        efficientnet.load_state_dict(weights)
        
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        in_features = efficientnet.classifier[1].in_features
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


def topk_accuracy(predictions, labels, k=5):
    """Calculate top-k accuracy"""
    topk_preds = predictions.topk(k, dim=1).indices
    correct = (topk_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
    return correct / len(labels)


@torch.inference_mode()
def evaluate(model, test_loader, device, label_names):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nRunning evaluation...")
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    
    # Top-k accuracies
    all_probs = torch.tensor(all_probs)
    all_labels_tensor = torch.tensor(all_labels)
    top1_acc = topk_accuracy(all_probs, all_labels_tensor, k=1)
    top3_acc = topk_accuracy(all_probs, all_labels_tensor, k=3)
    top5_acc = topk_accuracy(all_probs, all_labels_tensor, k=5)
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Test samples: {len(all_labels)}")
    print()
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Micro F1:      {micro_f1:.4f}")
    print(f"Top-1 Acc:     {top1_acc:.4f}")
    print(f"Top-3 Acc:     {top3_acc:.4f}")
    print(f"Top-5 Acc:     {top5_acc:.4f}")
    print()
    
    # Per-class F1 scores
    print("Per-Type Performance:")
    print("-"*80)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    
    for i, (type_name, f1) in enumerate(zip(label_names, per_class_f1)):
        num_samples = all_labels.count(i)
        print(f"  {type_name:12} F1={f1:.4f}  ({num_samples:4} samples)")
    
    # Confusion matrix analysis
    print()
    print("Most Common Mistakes:")
    print("-"*80)
    
    # Only compute confusion matrix if we have predictions for all classes
    unique_labels = set(all_labels)
    unique_preds = set(all_preds)
    
    if len(unique_labels) > 0 and len(unique_preds) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        
        mistakes = []
        # Only iterate over classes that exist in the data
        actual_num_classes = cm.shape[0]
        for true_idx in range(actual_num_classes):
            for pred_idx in range(cm.shape[1]):
                if true_idx != pred_idx and cm[true_idx][pred_idx] > 0:
                    # Map back to label names safely
                    true_label = all_labels[all_labels.index(true_idx)] if true_idx in all_labels else true_idx
                    pred_label = all_preds[all_preds.index(pred_idx)] if pred_idx in all_preds else pred_idx
                    
                    true_name = label_names[true_idx] if true_idx < len(label_names) else f"Class_{true_idx}"
                    pred_name = label_names[pred_idx] if pred_idx < len(label_names) else f"Class_{pred_idx}"
                    
                    mistakes.append((
                        cm[true_idx][pred_idx],
                        true_name,
                        pred_name
                    ))
        
        if mistakes:
            mistakes.sort(reverse=True)
            for count, true_type, pred_type in mistakes[:10]:
                print(f"  {true_type:12} → {pred_type:12} ({count:3} times)")
        else:
            print("  No mistakes! Perfect predictions!")
    else:
        print("  Not enough data to show mistakes")
    
    print("="*80)
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'top1_acc': top1_acc,
        'top3_acc': top3_acc,
        'top5_acc': top5_acc,
        'per_class_f1': per_class_f1.tolist()
    }


def main():
    args = parse_args()
    
    print("="*80)
    print("UNSEEN POKEMON EVALUATION")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Load checkpoint
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    label_names = checkpoint.get('label_names', [])
    num_classes = len(label_names)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_names}")
    print()
    
    # Build model
    model = ImprovedClassifier(num_classes=num_classes, dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load test dataset
    print("\nLoading test dataset...")
    
    # Create temporary structure for PokemonDataset
    test_data_root = Path(args.test_dir).parent
    
    # Create a custom dataset for testing
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    
    class TestDataset(Dataset):
        def __init__(self, test_dir, label_names):
            self.samples = []
            self.label_names = label_names
            self.label_to_idx = {name: i for i, name in enumerate(label_names)}
            
            # Find all images in type folders
            for type_folder in Path(test_dir).iterdir():
                if type_folder.is_dir() and type_folder.name in self.label_to_idx:
                    type_idx = self.label_to_idx[type_folder.name]
                    for ext in ['*.png', '*.jpg', '*.jpeg']:
                        for img_path in type_folder.glob(ext):
                            self.samples.append((str(img_path), type_idx))
            
            # Transforms
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path)
            img = self.transform(img)
            return img, label
    
    test_dataset = TestDataset(args.test_dir, label_names)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ Loaded {len(test_dataset)} test images")
    
    # Evaluate
    results = evaluate(model, test_loader, args.device, label_names)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()