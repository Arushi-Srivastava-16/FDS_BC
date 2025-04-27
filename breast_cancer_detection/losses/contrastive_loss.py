# breast_cancer_detection/losses/contrastive_loss.py

import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i, z_j: augmented embeddings [batch_size, feature_dim]
        """
        batch_size = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)  # [2*batch_size, feature_dim]
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.t())  # [2*batch_size, 2*batch_size]
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        mask = torch.eye(labels.shape[0], device=labels.device).bool()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)

        return loss
