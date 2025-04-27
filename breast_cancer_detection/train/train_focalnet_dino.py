# breast_cancer_detection/train_focalnet_dino.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.focalnet_dino import FocalNetDINO
from datasets.vindr_dataset import VinDrDataset
from configs.config import Config
from torchvision import transforms
import os

class FocalNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_boxes, gt_boxes):
        """
        pred_boxes: [batch_size, NUM_QUERIES, 4]
        gt_boxes: [batch_size, 4]
        """
        # We assume matching first predicted box to ground truth for now
        pred = pred_boxes[:, 0, :]  # taking the first proposal
        return self.mse_loss(pred, gt_boxes)

def train_focalnet_dino():
    device = Config.DEVICE

    # Model
    focalnet_dino = FocalNetDINO(num_queries=Config.NUM_QUERIES).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(focalnet_dino.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Loss
    loss_fn = FocalNetLoss()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset = VinDrDataset(Config.IMAGE_DIR, Config.ANNOTATIONS_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        focalnet_dino.train()
        running_loss = 0.0

        for batch in dataloader:
            images = batch['image'].to(device)
            gt_boxes = batch['boxes'].to(device)  # assuming 'boxes' is [batch_size, 4]

            optimizer.zero_grad()

            pred_boxes = focalnet_dino(images)

            loss = loss_fn(pred_boxes, gt_boxes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[FocalNetDINO] Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(Config.SAVE_MODEL_PATH, f"focalnet_dino_epoch{epoch+1}.pth")
            torch.save(focalnet_dino.state_dict(), save_path)

if __name__ == "__main__":
    train_focalnet_dino()
