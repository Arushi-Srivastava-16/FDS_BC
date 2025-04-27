# breast_cancer_detection/models/bayesian_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianHead(nn.Module):
    def __init__(self, input_dim=256):
        super(BayesianHead, self).__init__()
        self.mean_layer = nn.Linear(input_dim, 1)
        self.logvar_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        std = torch.exp(0.5 * logvar)
        return mean, std
