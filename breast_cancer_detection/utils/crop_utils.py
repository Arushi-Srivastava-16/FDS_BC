import torch
import torchvision.transforms as T
from PIL import Image
import os

# Transform to resize each crop to 224x224
resize_crop = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def extract_crops_from_predictions(image, pred_boxes, num_crops=5):
    """
    Extract and resize crops from normalized predicted bounding boxes.

    Args:
        image (PIL.Image or torch.Tensor): Original image.
        pred_boxes (torch.Tensor): [NUM_QUERIES, 4] with (x1, y1, x2, y2) normalized [0–1].
        num_crops (int): Number of crops to extract.

    Returns:
        List[torch.Tensor]: List of crops resized to [3, 224, 224].
    """
    crops = []

    # Convert tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        image = T.ToPILImage()(image.cpu())

    W, H = image.size

    # Make sure pred_boxes is on CPU
    pred_boxes = pred_boxes.detach().cpu()

    for i in range(min(num_crops, pred_boxes.size(0))):
        # Denormalize box coordinates
        x1, y1, x2, y2 = pred_boxes[i]
        x1 = int(x1.item() * W)
        y1 = int(y1.item() * H)
        x2 = int(x2.item() * W)
        y2 = int(y2.item() * H)

        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        # Skip invalid boxes
        if x2 > x1 and y2 > y1:
            crop = image.crop((x1, y1, x2, y2))
            crop = resize_crop(crop)  # Resize to 224x224 and convert to tensor
            crops.append(crop)

    return crops