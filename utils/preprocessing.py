from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights


class RGBAToRGB:
    """Convert RGBA images to RGB with a white background."""
    def __call__(self, img):
        if img.mode == "RGBA":
            rgb = Image.new("RGB", img.size, (255, 255, 255))
            rgb.paste(img, mask=img.split()[3])
            return rgb
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def get_val_transform(weights=None):
    """Preprocessing only — no augmentation. Use for val/test and inference."""
    weights = weights or EfficientNet_B0_Weights.DEFAULT
    effnet_preprocess = weights.transforms()
    return transforms.Compose([
        RGBAToRGB(),
        effnet_preprocess,
    ])


def get_train_transform(weights=None):
    """
    Augmented preprocessing for training data only.

    Augmentations applied (all on PIL before EfficientNet resize/normalize):
      - RandomHorizontalFlip      (p=0.5)
      - RandomRotation            (±30°, white fill to match background)
      - ColorJitter               (brightness/contrast ±30%, saturation ±10%)
      - RandomAffine              (translate 15%, scale 80-120%, white fill)

    fill=255 keeps the background white, matching what RGBAToRGB produces.
    """
    weights = weights or EfficientNet_B0_Weights.DEFAULT
    effnet_preprocess = weights.transforms()
    return transforms.Compose([
        RGBAToRGB(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30, fill=255),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), fill=255),
        effnet_preprocess,
    ])


def get_offline_augmentation():
    """
    PIL-only augmentation pipeline for saving augmented images to disk.
    More aggressive than online augmentation to maximise image diversity.
    Returns a transform that takes a PIL image and returns a PIL image.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(degrees=35, fill=255),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.15, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), fill=255),
    ])


# Backwards-compatible alias
def get_transform(weights=None):
    return get_val_transform(weights)