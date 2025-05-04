# SimCLR architecture + paired dataset setup

import os
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# -----------------------------
# SimCLR Projection Head
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# SimCLR Encoder (ResNet50 + Head)
# -----------------------------
class SimCLR(nn.Module):
    def __init__(self, base_model='resnet50', projection_dim=128):
        super().__init__()
        resnet = models.resnet50(weights=None)  # weights='IMAGENET1K_V1' for pre-trained
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
        self.projection_head = ProjectionHead(input_dim=2048, projection_dim=projection_dim)

    def forward(self, x):
        features = self.encoder(x).squeeze()
        projections = self.projection_head(features)
        return projections

# -----------------------------
# Paired Dataset for SimCLR
# -----------------------------
class PairedSimCLRDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.pairs = self._create_positive_pairs()

    def _create_positive_pairs(self):
        grouped = self.data.groupby(['study_id', 'laterality'])
        pairs = []
        for _, group in grouped:
            cc = group[group['view'] == 'CC']
            mlo = group[group['view'] == 'MLO']
            for _, cc_row in cc.iterrows():
                for _, mlo_row in mlo.iterrows():
                    pairs.append((cc_row['filename'], mlo_row['filename']))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cc_path = os.path.join(self.root_dir, self._folder(cc_path := self.pairs[idx][0]), cc_path)
        mlo_path = os.path.join(self.root_dir, self._folder(mlo_path := self.pairs[idx][1]), mlo_path)

        cc_img = Image.open(cc_path).convert('RGB')
        mlo_img = Image.open(mlo_path).convert('RGB')

        if self.transform:
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)

        return cc_img, mlo_img

    def _folder(self, fname):
        return 'positive' if 'positive' in fname else 'negative'

# -----------------------------
# Define transforms and dataloader
# -----------------------------
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset & Loader
csv_path = 'crop_metadata.csv'
root_dir = 'cropped_images_simclr'
dataset = PairedSimCLRDataset(csv_path, root_dir, transform=data_transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# SimCLR Model
model = SimCLR()

# Sample usage
for x1, x2 in dataloader:
    out1 = model(x1)
    out2 = model(x2)
    print("Batch outputs:", out1.shape, out2.shape)
    break