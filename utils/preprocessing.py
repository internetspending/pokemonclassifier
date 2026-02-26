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


def get_transform(weights=None):
    """
    Preprocessing pipeline for EfficientNet-B0 pretrained weights.
    - Keep RGBA->RGB fix
    - Use weights.transforms() for correct resize/crop/normalize
    """
    weights = weights or EfficientNet_B0_Weights.DEFAULT
    effnet_preprocess = weights.transforms()
    return transforms.Compose([
        RGBAToRGB(),
        effnet_preprocess,
    ])