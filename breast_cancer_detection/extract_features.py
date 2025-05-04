import os
import torch
from PIL import Image
from torchvision import transforms
from Faltu.backbone import FeatureExtractor
from tqdm import tqdm
import numpy as np

# Directory where crops are saved
CROP_DIR = "crops/"
SAVE_FEATURES_PATH = "crop_features.npy"  # Optional: save as CSV if needed

# Transform (must match model input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FeatureExtractor(model_type='resnet50', out_dim=256).to(device)
model.eval()

all_embeddings = []
image_names = []

# Extract features
with torch.no_grad():
    for fname in tqdm(os.listdir(CROP_DIR)):
        if not fname.endswith(".png"):
            continue
        img_path = os.path.join(CROP_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        embedding = model(img_tensor).cpu().numpy()[0]  # Shape: (256,)
        all_embeddings.append(embedding)
        image_names.append(fname)

# Save features
np.save(SAVE_FEATURES_PATH, np.array(all_embeddings))
with open("feature_index.txt", "w") as f:
    for name in image_names:
        f.write(f"{name}\n")

print("✅ Done. Saved feature matrix to", SAVE_FEATURES_PATH)