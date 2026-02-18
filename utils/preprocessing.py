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


def get_transform(weights=None, train=False):
    """
    Preprocessing pipeline for EfficientNet-B0 pretrained weights.
    - Keep RGBA->RGB fix
    - Use weights.transforms() for correct resize/crop/normalize
    - if train is True, apply data augmentation
    """
    weights = weights or EfficientNet_B0_Weights.DEFAULT
    effnet_preprocess = weights.transforms()

    if train: 
        # Training: with augmentation
        return transforms.Compose([
            RGBAToRGB(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            RGBAToRGB(),
            effnet_preprocess,
        ])