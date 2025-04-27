# breast_cancer_detection/train_contrastive.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.backbone import Backbone
from losses.contrastive_loss import NTXentLoss
from datasets.vindr_dataset import VinDrDataset
from torchvision import transforms
from configs.config import Config
import os

def train_contrastive():
    device = Config.DEVICE
    backbone = Backbone(name=Config.BACKBONE).to(device)
    projection_head = nn.Sequential(
        nn.Linear(backbone.out_dim, Config.CONTRASTIVE_PROJECTION_DIM),
        nn.ReLU(),
        nn.Linear(Config.CONTRASTIVE_PROJECTION_DIM, Config.CONTRASTIVE_PROJECTION_DIM)
    ).to(device)

    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projection_head.parameters()), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    contrastive_loss_fn = NTXentLoss(temperature=Config.TEMPERATURE)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = VinDrDataset(Config.IMAGE_DIR, Config.ANNOTATIONS_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        backbone.train()
        projection_head.train()
        running_loss = 0.0

        for batch in dataloader:
            x = batch['image'].to(device)

            x_i = transforms.RandomApply([transform])(x)
            x_j = transforms.RandomApply([transform])(x)

            optimizer.zero_grad()

            h_i = backbone(x_i).squeeze()
            h_j = backbone(x_j).squeeze()

            z_i = projection_head(h_i)
            z_j = projection_head(h_j)

            loss = contrastive_loss_fn(z_i, z_j)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(backbone.state_dict(), os.path.join(Config.SAVE_MODEL_PATH, f"backbone_epoch{epoch+1}.pth"))

if __name__ == "__main__":
    train_contrastive()
