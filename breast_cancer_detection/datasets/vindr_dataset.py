import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class VinDrDataset(Dataset):
    def __init__(self, image_dir, annotation_csv, transform=None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotation_csv)
        self.transform = transform

        # Filter rows that have a valid bounding box
        self.annotations['has_box'] = ~(
            (self.annotations['xmin'] == 0) & 
            (self.annotations['ymin'] == 0) & 
            (self.annotations['xmax'] == 0) & 
            (self.annotations['ymax'] == 0)
        )

        # Only keep rows where bounding boxes are valid
        self.annotations = self.annotations[self.annotations['has_box'] == True].reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        if self.transform:
            image = self.transform(image)

        # Normalize bounding box to [0, 1]
        box = torch.tensor([
            row['xmin'] / width,
            row['ymin'] / height,
            row['xmax'] / width,
            row['ymax'] / height
        ], dtype=torch.float32)

        return {
            'image': image,
            'boxes': box
        }
