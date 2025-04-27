# breast_cancer_detection/utils/crop_utils.py

import torch
import torchvision.transforms.functional as TF

def crop_regions(image, boxes, crop_size=(128, 128)):
    """
    Crops regions from the image based on bounding boxes.

    Args:
        image (Tensor): [C, H, W]
        boxes (Tensor): [N, 5]  (x, y, w, h, confidence)
        crop_size (tuple): output size (H, W) for each crop

    Returns:
        crops (Tensor): [N, C, crop_size[0], crop_size[1]]
    """
    crops = []
    _, H, W = image.shape
    for box in boxes:
        x, y, w, h, conf = box
        x1 = int(max(x, 0))
        y1 = int(max(y, 0))
        x2 = int(min(x + w, W))
        y2 = int(min(y + h, H))
        crop = image[:, y1:y2, x1:x2]
        crop = TF.resize(crop, crop_size)
        crops.append(crop)
    crops = torch.stack(crops) if crops else torch.empty(0)
    return crops
