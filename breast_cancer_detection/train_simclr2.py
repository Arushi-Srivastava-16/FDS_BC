import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.simclr_model import SimCLR
from datasets.simclr_dataset import PairedSimCLRDataset  # Ensure correct import
from tqdm import tqdm

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z1, z2):
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2N, D]
        z = nn.functional.normalize(z, dim=1)  # L2 normalization

        # Cosine similarity matrix: [2N, 2N]
        similarity_matrix = torch.matmul(z, z.T)  # cosine sim since z is normalized
        # Remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))

        # Positive pairs: (i, i+N) and (i+N, i)
        positives = torch.cat([torch.arange(batch_size, device=z.device) + batch_size,
                               torch.arange(batch_size, device=z.device)], dim=0)

        logits = similarity_matrix / self.temperature
        labels = positives

        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = PairedSimCLRDataset(csv_path='crop_metadata.csv', root_dir='cropped_images_simclr', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # use num_workers=0 if still error

    model = SimCLR().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss(temperature=0.5)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for x1, x2 in loop:
            x1, x2 = x1.to(device), x2.to(device)

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

        # Save model every 2 epochs
        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f"simclr_paired_epoch_{epoch+1}.pt")
            print(f"Model checkpoint saved at epoch {epoch+1}")