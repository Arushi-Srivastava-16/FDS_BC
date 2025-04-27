# breast_cancer_detection/models/protopnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoPNet(nn.Module):
    def __init__(self, feature_dim=256, num_prototypes=10):
        super(ProtoPNet, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))

    def forward(self, features):
        """
        Computes similarity between input features and learned prototypes.

        Args:
            features: [B, D]
        Returns:
            similarity: [B, num_prototypes]
        """
        features = F.normalize(features, dim=1)
        prototypes = F.normalize(self.prototypes, dim=1)
        similarity = torch.matmul(features, prototypes.t())
        return similarity
