import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import csv
from models.simclr_model import SimCLR

# --- Classifier definition (same as used in training) ---
class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim=2048):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()  # Ensure projection head is not used
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

# --- Inference Dataset ---
class InferenceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = []
        self.labels = []  # Optional: dummy labels if you want to compare
        self.transform = transform
        
        for label, subfolder in enumerate(['negative', 'positive']):
            full_path = os.path.join(folder_path, subfolder)
            if os.path.exists(full_path):
                for fname in os.listdir(full_path):
                    fpath = os.path.join(full_path, fname)
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.image_paths.append(fpath)
                        self.labels.append(label)
            else:
                # If subfolders don't exist, use images directly
                for fname in os.listdir(folder_path):
                    fpath = os.path.join(folder_path, fname)
                    if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                        self.image_paths.append(fpath)
                        self.labels.append(-1)  # Unknown label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path, self.labels[idx]

# --- Inference Function ---
def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset and loader
    dataset = InferenceDataset(args.input_dir, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load encoder
    encoder = SimCLR().encoder
    state_dict = torch.load(args.encoder_weights, map_location=device)
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.") and "projection_head" not in k
    }
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # Load full classifier model
    model = Classifier(encoder).to(device)
    model.load_state_dict(torch.load(args.classifier_weights, map_location=device))
    model.eval()

    sigmoid = nn.Sigmoid()

    results = []
    with torch.no_grad():
        for images, paths, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = sigmoid(logits).squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)

            for path, prob, pred, label in zip(paths, probs, preds, labels):
                results.append({
                    "image": path,
                    "predicted_label": pred,
                    "confidence": float(prob),
                    "true_label": int(label) if label != -1 else "N/A"
                })
                print(f"{os.path.basename(path)} -> Pred: {pred} (Conf: {prob:.4f})")

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output_csv}")

# --- CLI entrypoint ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR Inference Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images or folder (can contain 'positive'/'negative' subfolders)")
    parser.add_argument("--encoder_weights", type=str, default="simclr_paired_epoch_22.pt", help="Path to trained SimCLR weights")
    parser.add_argument("--classifier_weights", type=str, default="simclr_classifier.pth", help="Path to trained classifier weights")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_csv", type=str, help="Optional path to save predictions as CSV")
    
    args = parser.parse_args()
    run_inference(args)