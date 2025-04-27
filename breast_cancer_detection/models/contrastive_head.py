# breast_cancer_detection/models/contrastive_head.py

import torch.nn as nn

class ContrastiveHead(nn.Module):
    def __init__(self, input_dim=2048, projection_dim=256):
        """
        Projection head used for contrastive learning (SimCLR style).
        """
        super(ContrastiveHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )

    def forward(self, x):
        return self.net(x)
