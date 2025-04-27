# breast_cancer_detection/datasets/vindr_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class VinDrDataset(Dataset):
    def __init__(self, image_dir, annotations=None, transform=None):
        """
        Args:
            image_dir (str): Path to directory containing PNG images.
            annotations (dict): {image_id: [list of bounding boxes]}.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Annotations (bounding boxes) if available
        boxes = None
        if self.annotations and img_name in self.annotations:
            boxes = torch.tensor(self.annotations[img_name], dtype=torch.float32)

        return {
            'image': image,
            'filename': img_name,
            'boxes': boxes   # (optional) could be None during inference
        }
