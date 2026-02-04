from PIL import Image
from torchvision import transforms


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


def get_transform():
    """Return the preprocessing pipeline for EfficientNet-B0."""
    return transforms.Compose([
        RGBAToRGB(),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])
