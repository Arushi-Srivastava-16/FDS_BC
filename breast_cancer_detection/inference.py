# breast_cancer_detection/inference.py

import torch
from torchvision import transforms
from models.focalnet_dino import DummyFocalNetDINO
from datasets.vindr_dataset import VinDrDataset
from utils.visualization import visualize_proposals
from configs.config import Config

def run_inference():
    model = DummyFocalNetDINO(num_queries=Config.NUM_QUERIES)
    model.load_state_dict(torch.load(Config.CHECKPOINT_PATH))
    model = model.to(Config.DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    dataset = VinDrDataset(Config.IMAGE_DIR, transform=transform)
    image = dataset[0]['image'].unsqueeze(0).to(Config.DEVICE)

    with torch.no_grad():
        proposals = model(image)[0]  # [num_queries, 5]

    visualize_proposals(image.squeeze(0), proposals.cpu())

if __name__ == "__main__":
    run_inference()
