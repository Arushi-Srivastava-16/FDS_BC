# breast_cancer_detection/models/refinement_network.py

import torch.nn as nn

class RefinementNetwork(nn.Module):
    def __init__(self, input_dim=256, output_dim=5):
        super(RefinementNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)
