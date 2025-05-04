import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ----------------------------
# Dataset class
# ----------------------------
class LabelledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, folder in enumerate(['positive']):
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
        return image, label, path

# ----------------------------
# Classifier wrapper
# ----------------------------
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
        features = self.encoder(x)
        return self.classifier(features)

# ----------------------------
# GradCAM utility
# ----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor):
        output = self.model(input_tensor)
        score = output[:, 0].sigmoid().mean()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cam

# ----------------------------
# Heatmap overlay utility
# ----------------------------
def overlay_heatmap(original_img, cam, save_path):
    cam = cv2.resize(cam, original_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.array(original_img) * 0.5 + heatmap * 0.5
    overlay = overlay / 255.0

    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ----------------------------
# Main script
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load encoder
    from models.simclr_model import SimCLR
    encoder = SimCLR().encoder
    encoder.load_state_dict({
        k.replace("encoder.", ""): v
        for k, v in torch.load("simclr_paired_epoch_22.pt", map_location=device).items()
        if k.startswith("encoder.") and "projection_head" not in k
    })
    encoder.eval()

    # Build model
    model = Classifier(encoder).to(device)
    model.load_state_dict(torch.load("bayesian_simclr_classifier.pth", map_location=device))
    model.eval()

    # Choose target layer (last conv layer of ResNet50)
    target_layer = model.encoder.layer4[-1].conv3
    gradcam = GradCAM(model, target_layer)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = LabelledDataset("cropped_images_simclr", transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs("gradcam_outputs", exist_ok=True)

    num_pos, num_neg = 0, 0
    max_per_class = 5  # change this number as you wish

    for img_tensor, label, path in loader:
        if num_pos >= max_per_class and num_neg >= max_per_class:
            break

        img_tensor = img_tensor.to(device)
        label = label.item()

        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

        if prob > 0.5 and num_pos < max_per_class:
            cam = gradcam.generate_heatmap(img_tensor)
            original_img = Image.open(path[0]).convert("RGB").resize((224, 224))
            filename = os.path.basename(path[0])
            save_path = os.path.join("gradcam_outputs", f"POS_cam_{filename}")
            overlay_heatmap(original_img, cam, save_path)
            num_pos += 1
            print(f"[POS] Saved: {save_path}")

        elif prob <= 0.5 and num_neg < max_per_class:
            cam = gradcam.generate_heatmap(img_tensor)
            original_img = Image.open(path[0]).convert("RGB").resize((224, 224))
            filename = os.path.basename(path[0])
            save_path = os.path.join("gradcam_outputs", f"NEG_cam_{filename}")
            overlay_heatmap(original_img, cam, save_path)
            num_neg += 1
            print(f"[NEG] Saved: {save_path}")

if __name__ == "__main__":
    main()