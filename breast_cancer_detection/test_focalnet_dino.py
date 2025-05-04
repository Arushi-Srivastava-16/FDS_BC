# test_focalnet_dino.py



# to run cause ik we forget :          python test_focalnet_dino.py \ --image path/to/test_image.png \--checkpoint path/to/focalnet_dino_checkpoint.pth

import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from models.focalnet_dino import FocalNetDINO
from configs.config import Config
import argparse
import os

# ---- Load model ----
def load_model(checkpoint_path):
    model = FocalNetDINO(num_queries=Config.NUM_QUERIES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    model.eval()
    return model

# ---- Visualize predictions ----
def visualize_predictions(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Load image
    image_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image_orig.size
    # Prepare input
    input_tensor = transform(image_orig).unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        pred_boxes = model(input_tensor)  # Shape: [1, NUM_QUERIES, 5]
        pred_boxes = pred_boxes[0].cpu()

    draw = ImageDraw.Draw(image_orig)

    for box in pred_boxes:
        x1, y1, x2, y2 = box[:4].tolist()
        # Convert normalized coords back to pixel space
        x1 *= orig_w
        x2 *= orig_w
        y1 *= orig_h
        y2 *= orig_h
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        print(x1,x2,y1,y2)
        draw.rectangle([736, 800, 888, 965], outline='green', width=3)

    image_orig.show()

# ---- Main Function ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    args = parser.parse_args()

    # Load model and run inference
    model = load_model(args.checkpoint)
    visualize_predictions(args.image, model)