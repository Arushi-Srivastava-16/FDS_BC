import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class Classifier(nn.Module):
    def __init__(self, encoder, feature_dim=2048):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.encoder.fc = nn.Identity()  # Remove final fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary classification
        )

    def forward(self, x):
        with torch.no_grad():  # Freeze encoder
            features = self.encoder(x)
        return self.classifier(features)

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare encoder
    from models.simclr_model import SimCLR  # Import your SimCLR implementation
    
    # Load state dict with security fix
    state_dict = torch.load("FDS_BC/breast_cancer_detection/SimCLR/simclr_paired_epoch_26.pt", 
                          map_location=device,
                          weights_only=True)  # Security fix
    
    # Modify state dict keys
    encoder_state_dict = {
        k.replace("encoder.", ""): v 
        for k, v in state_dict.items()
        if k.startswith("encoder.") and "projection_head" not in k
    }
    
    # Initialize encoder
    encoder = SimCLR().encoder
    encoder.load_state_dict(encoder_state_dict)
    encoder.eval()
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Prepare data
    train_data = LabelledDataset("cropped_images_simclr", transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    
    # Initialize model
    model = Classifier(encoder).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(30):
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().to(device).unsqueeze(1)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"simcsimclr_classifier_{epoch+1}.pth")
            print(f"Model checkpoint saved at epoch {epoch+1}")
    
    # Save trained classifier
    torch.save(model.state_dict(), "simclr_classifier.pth")
    

if __name__ == "__main__":
    main()