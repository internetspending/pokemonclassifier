import sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 18

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] GPU = {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU (this is fine; same code will use GPU on a CUDA setup)")

    # 1) Load pretrained weights + matching preprocessing
    print("[STEP] Loading pretrained EfficientNet-B0 weights + preprocessing transforms...")
    weights = EfficientNet_B0_Weights.DEFAULT
    preprocess = weights.transforms()

    # 2) Build model
    print("[STEP] Building model...")
    model = efficientnet_b0(weights=weights)

    # 3) Replace classifier (1000 -> 18)
    old_out = model.classifier[1].out_features
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    print(f"[STEP] Replaced classifier: {old_out} -> {NUM_CLASSES} outputs")

    model = model.to(device).eval()

    # 4) Run inference (real image if provided; otherwise dummy tensor)
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        print(f"[RUN] Image inference on: {img_path}")
        img = Image.open(img_path).convert("RGB")

        x = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        print(f"[INFO] input shape = {tuple(x.shape)}")

        with torch.inference_mode():
            logits = model(x)

        print(f"[RESULT] output shape = {tuple(logits.shape)} (expected: (1, {NUM_CLASSES}))")

        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=min(5, NUM_CLASSES))
        print("[RESULT] top predictions (class_index, prob):")
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            print(f"  {idx:>2}  {p:.4f}")

    else:
        print("[RUN] Dummy forward pass (no image provided)")
        x = torch.randn(1, 3, 224, 224, device=device)
        print(f"[INFO] input shape = {tuple(x.shape)}")

        with torch.inference_mode():
            y = model(x)

        print(f"[RESULT] output shape = {tuple(y.shape)} (expected: (1, {NUM_CLASSES}))")

if __name__ == "__main__":
    main()
