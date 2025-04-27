# breast_cancer_detection/train_full.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.backbone import Backbone
from models.focalnet_dino import FocalNetDINO
from models.refinement_network import RefinementMLP
from models.contrastive_head import ProjectionHead
from models.protopnet import ProtoPNet
from models.bayesian_head import BayesianHead
from datasets.vindr_dataset import VinDrDataset
from losses.contrastive_loss import NTXentLoss
from configs.config import Config
from utils.crop_utils import extract_crops
from torchvision import transforms
import os

def train_full_model():
    device = Config.DEVICE

    # Models
    backbone = Backbone(name=Config.BACKBONE).to(device)
    focalnet_dino = FocalNetDINO(num_queries=Config.NUM_QUERIES).to(device)
    refinement_mlp = RefinementMLP(input_dim=256).to(device)
    projection_head = ProjectionHead(backbone_dim=backbone.out_dim, projection_dim=Config.CONTRASTIVE_PROJECTION_DIM).to(device)
    protopnet = ProtoPNet(num_prototypes=Config.NUM_QUERIES, feature_dim=Config.CONTRASTIVE_PROJECTION_DIM).to(device)
    bayesian_head = BayesianHead(input_dim=Config.CONTRASTIVE_PROJECTION_DIM).to(device)

    # Optimizer
    all_params = list(backbone.parameters()) + \
                 list(focalnet_dino.parameters()) + \
                 list(refinement_mlp.parameters()) + \
                 list(projection_head.parameters()) + \
                 list(protopnet.parameters()) + \
                 list(bayesian_head.parameters())

    optimizer = torch.optim.Adam(all_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    # Loss Functions
    contrastive_loss_fn = NTXentLoss(temperature=Config.TEMPERATURE)
    classification_loss_fn = nn.CrossEntropyLoss()
    uncertainty_loss_fn = nn.MSELoss()  # for Bayesian uncertainty

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset = VinDrDataset(Config.IMAGE_DIR, Config.ANNOTATIONS_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)

    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)

    for epoch in range(Config.NUM_EPOCHS):
        backbone.train()
        focalnet_dino.train()
        refinement_mlp.train()
        projection_head.train()
        protopnet.train()
        bayesian_head.train()

        running_loss = 0.0

        for batch in dataloader:
            images = batch['image'].to(device)

            optimizer.zero_grad()

            # Step 1: Detect regions (coordinates) using FocalNet-DINO
            pred_coords = focalnet_dino(images)  # [batch_size, NUM_QUERIES, 4]

            # Step 2: Crop regions from images
            crops = extract_crops(images, pred_coords)

            # Step 3: Pass crops through Backbone
            crop_features = backbone(crops)

            # Step 4: Refine proposals
            refined_features = refinement_mlp(crop_features)

            # Step 5: Contrastive learning
            projected_features = projection_head(refined_features)

            # Step 6: ProtoPNet for classification
            class_logits = protopnet(projected_features)

            # Step 7: Bayesian head for uncertainty estimation
            uncertainties = bayesian_head(projected_features)

            # Losses
            contrastive_loss = contrastive_loss_fn(projected_features, projected_features)
            classification_loss = classification_loss_fn(class_logits, batch['label'].to(device))  # assuming labels exist
            uncertainty_target = torch.zeros_like(uncertainties).to(device)  # placeholder target
            uncertainty_loss = uncertainty_loss_fn(uncertainties, uncertainty_target)

            total_loss = contrastive_loss + classification_loss + 0.1 * uncertainty_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Loss: {running_loss/len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(Config.SAVE_MODEL_PATH, f"full_model_epoch{epoch+1}.pth")
            torch.save({
                'backbone': backbone.state_dict(),
                'focalnet_dino': focalnet_dino.state_dict(),
                'refinement_mlp': refinement_mlp.state_dict(),
                'projection_head': projection_head.state_dict(),
                'protopnet': protopnet.state_dict(),
                'bayesian_head': bayesian_head.state_dict(),
            }, save_path)

if __name__ == "__main__":
    train_full_model()
