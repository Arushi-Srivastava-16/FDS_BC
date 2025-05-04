import torch
from PIL import Image
from torchvision import transforms as T
from utils.crop_utils import extract_crops_from_predictions

# Load a sample image (PNG, RGB)
image_path = 'vindrpng/images_png/0c74dbdbe72ae034c4a429da7b60f55d/15988057d6e70ed0751384b82dd7f2d4.png'
image = Image.open(image_path).convert("RGB")

# Dummy predicted boxes (normalized [0,1])
# Format: (x1, y1, x2, y2) in normalized coords
pred_boxes = torch.tensor([
    [0.1, 0.1, 0.5, 0.5],
    [0.3, 0.3, 0.7, 0.7],
    [0.0, 0.0, 0.2, 0.2],
])

# Get crops
crops = extract_crops_from_predictions(image, pred_boxes, num_crops=3)

# Save crops to disk for verification
for i, crop in enumerate(crops):
    crop_image = T.ToPILImage()(crop)
    crop_image.save(f"./crop_{i+1}.png")

print(f"Saved {len(crops)} crop(s).")