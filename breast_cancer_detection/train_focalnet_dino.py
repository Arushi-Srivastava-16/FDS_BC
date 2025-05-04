import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.focalnet_dino import FocalNetDINO
from datasets.vindr_dataset import VinDrDataset
from configs.config import Config

class FocalNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_boxes, gt_boxes):
        """
        pred_boxes: [batch_size, NUM_QUERIES, 5]
        gt_boxes: [batch_size, 4]
        """
        pred = pred_boxes[:, 0, :4]  # take only x, y, w, h
        return self.mse_loss(pred, gt_boxes)

def train_focalnet_dino():
    device = Config.DEVICE

    # Initialize model
    model = FocalNetDINO(num_queries=Config.NUM_QUERIES).to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    loss_fn = FocalNetLoss()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    dataset = VinDrDataset(Config.IMAGE_DIR, Config.ANNOTATIONS_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            images = batch['image'].to(device)
            gt_boxes = batch['boxes'].to(device)

            optimizer.zero_grad()
            pred_boxes = model(images)
            loss = loss_fn(pred_boxes, gt_boxes)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch+1}/{Config.NUM_EPOCHS}] Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(Config.SAVE_MODEL_PATH, f"focalnet_dino_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    train_focalnet_dino()