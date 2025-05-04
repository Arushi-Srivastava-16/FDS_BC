import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import numpy as np

# ----- Dataset -----
class LabelledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, folder in enumerate(['negative', 'positive']):
            full_path = os.path.join(root_dir, folder)
            for file in os.listdir(full_path):
                self.samples.append((os.path.join(full_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ----- Classifier -----
class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim=2048):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()
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

# ----- Evaluation Script -----
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SimCLR encoder
    from models.simclr_model import SimCLR
    encoder = SimCLR().encoder
    state_dict = torch.load("FDS_BC/breast_cancer_detection/SimCLR/simclr_paired_epoch_26.pt", map_location=device)
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.") and "projection_head" not in k
    }
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()

    # Initialize model and load classifier weights
    model = Classifier(encoder).to(device)
    model.load_state_dict(torch.load("FDS_BC/breast_cancer_detection/SimCLR/simclr_classifier.pth", map_location=device))
    model.eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    test_data = LabelledDataset("cropped_images_simclr", transform=transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            y_pred.extend((probs > 0.5).int().cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    y_prob = np.array(y_prob).flatten()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.close()

    # Plot Precision-Recall curve
    precs, recs, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recs, precs)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")
    plt.close()

if __name__ == "__main__":
    evaluate()