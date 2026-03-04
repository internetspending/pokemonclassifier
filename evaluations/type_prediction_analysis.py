"""
Secondary Type Prediction Analysis

For each dual-type Pokemon in the test set, categorizes the model's top-1
prediction as:
  - "primary hit"   — predicted Type1
  - "secondary hit" — predicted Type2 (learned without explicit supervision)
  - "miss"          — predicted neither

Also reports:
  - Mono-type vs dual-type Pokemon accuracy
  - Per-type secondary hit counts (which types does the model pick up visually?)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from sklearn.model_selection import train_test_split

from utils.dataset import PokemonDataset
from utils.preprocessing import get_val_transform
from torch.utils.data import Subset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn


def get_split_indices(dataset, seed=42):
    labels = [type1_idx for _, type1_idx, _ in dataset.samples]
    indices = list(range(len(dataset)))
    trainval_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=labels
    )
    return test_idx


def load_model(ckpt_path, num_classes, device):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def analyze(model, test_idx, dataset, device):
    results = []  # (pokemon_name, type1, type2_or_None, prediction, outcome)

    with torch.inference_mode():
        for orig_idx in test_idx:
            img_path, type1_idx, type2_idx = dataset.samples[orig_idx]
            img, _ = dataset[orig_idx]
            logits = model(img.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).item()

            name = os.path.splitext(os.path.basename(img_path))[0]
            type1_name = dataset.label_names[type1_idx]
            type2_name = dataset.label_names[type2_idx] if type2_idx is not None else None

            if pred == type1_idx:
                outcome = "primary_hit"
            elif type2_idx is not None and pred == type2_idx:
                outcome = "secondary_hit"
            else:
                outcome = "miss"

            results.append((name, type1_name, type2_name, dataset.label_names[pred], outcome))

    return results


def print_report(results, label_names):
    dual = [r for r in results if r[2] is not None]
    mono = [r for r in results if r[2] is None]

    # --- Overall breakdown for dual-type Pokemon ---
    counts = defaultdict(int)
    for r in dual:
        counts[r[4]] += 1

    print("=" * 55)
    print("DUAL-TYPE POKEMON — TOP-1 PREDICTION BREAKDOWN")
    print("=" * 55)
    total_dual = len(dual)
    for outcome in ("primary_hit", "secondary_hit", "miss"):
        n = counts[outcome]
        print(f"  {outcome:<15} {n:>4}  ({100*n/total_dual:.1f}%)")
    print(f"  {'total':<15} {total_dual:>4}")

    # --- Mono vs dual accuracy ---
    print("\n" + "=" * 55)
    print("MONO-TYPE vs DUAL-TYPE ACCURACY")
    print("=" * 55)
    mono_acc = sum(1 for r in mono if r[4] == "primary_hit") / len(mono) if mono else 0
    dual_acc = sum(1 for r in dual if r[4] in ("primary_hit", "secondary_hit")) / total_dual if dual else 0
    print(f"  Mono-type accuracy (type1 hit):          {mono_acc:.3f}  ({len(mono)} pokemon)")
    print(f"  Dual-type accuracy (type1 OR type2 hit): {dual_acc:.3f}  ({total_dual} pokemon)")

    # --- Per-type secondary hit breakdown ---
    secondary_hits_by_type = defaultdict(int)
    secondary_opportunities_by_type = defaultdict(int)
    for r in dual:
        secondary_opportunities_by_type[r[2]] += 1
        if r[4] == "secondary_hit":
            secondary_hits_by_type[r[2]] += 1

    print("\n" + "=" * 55)
    print("SECONDARY HITS BY TYPE (types model learned visually)")
    print("=" * 55)
    rows = []
    for t in sorted(secondary_opportunities_by_type.keys()):
        hits = secondary_hits_by_type[t]
        opps = secondary_opportunities_by_type[t]
        rows.append((t, hits, opps))
    rows.sort(key=lambda x: -x[1])
    for t, hits, opps in rows:
        bar = "#" * hits
        print(f"  {t:<10} {hits:>2}/{opps:<3} {bar}")

    # --- Examples of secondary hits ---
    sec_examples = [(r[0], r[1], r[2], r[3]) for r in results if r[4] == "secondary_hit"]
    if sec_examples:
        print("\n" + "=" * 55)
        print("SECONDARY HIT EXAMPLES")
        print("=" * 55)
        print(f"  {'Pokemon':<22} {'Type1':<10} {'Type2':<10} {'Predicted'}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
        for name, t1, t2, pred in sec_examples[:20]:
            print(f"  {name:<22} {t1:<10} {t2:<10} {pred}")


def plot_results(results):
    dual = [r for r in results if r[2] is not None]
    counts = defaultdict(int)
    for r in dual:
        counts[r[4]] += 1

    # --- Pie chart: primary / secondary / miss ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Secondary Type Prediction Analysis", fontsize=14, fontweight="bold")

    labels = ["Primary hit\n(Type1)", "Secondary hit\n(Type2)", "Miss"]
    sizes = [counts["primary_hit"], counts["secondary_hit"], counts["miss"]]
    colors = ["#4CAF50", "#2196F3", "#F44336"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Dual-type Pokemon: where does credit come from?")

    # --- Bar chart: secondary hits per type ---
    secondary_hits_by_type = defaultdict(int)
    secondary_opportunities_by_type = defaultdict(int)
    for r in dual:
        secondary_opportunities_by_type[r[2]] += 1
        if r[4] == "secondary_hit":
            secondary_hits_by_type[r[2]] += 1

    types_with_hits = sorted(
        [t for t in secondary_opportunities_by_type if secondary_hits_by_type[t] > 0],
        key=lambda t: -secondary_hits_by_type[t]
    )
    if types_with_hits:
        hit_counts = [secondary_hits_by_type[t] for t in types_with_hits]
        opp_counts = [secondary_opportunities_by_type[t] for t in types_with_hits]

        x = range(len(types_with_hits))
        axes[1].bar(x, opp_counts, color="#BBDEFB", label="Opportunities (as Type2)")
        axes[1].bar(x, hit_counts, color="#2196F3", label="Secondary hits")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(types_with_hits, rotation=45, ha="right")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Which secondary types does the model learn visually?")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No secondary hits", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("Which secondary types does the model learn visually?")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "type_prediction_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.show()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = os.path.join(PROJECT_ROOT, "data")
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "best_pokemon_classifier.pt")

    dataset = PokemonDataset(data_dir=data_dir, transform=get_val_transform())
    test_idx = get_split_indices(dataset)

    print(f"Test set size: {len(test_idx)}")
    print(f"Loading checkpoint: {ckpt_path}\n")

    model = load_model(ckpt_path, num_classes=len(dataset.label_names), device=device)
    results = analyze(model, test_idx, dataset, device)

    print_report(results, dataset.label_names)
    plot_results(results)


if __name__ == "__main__":
    main()
