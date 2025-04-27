# breast_cancer_detection/models/focalnet_dino.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalNetDINO(nn.Module):
    def __init__(self, num_queries=100, hidden_dim=256):
        super(FocalNetDINO, self).__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Tiny FocalNet/Transformer Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=3)

        # Heads
        self.bbox_head = nn.Linear(hidden_dim, 4)   # (x, y, w, h)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.flatten(2).permute(2, 0, 1)  # [HW, B, C]

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, B, C]
        transformer_out = self.transformer_encoder(queries + x.mean(0, keepdim=True))  # [num_queries, B, C]

        transformer_out = transformer_out.permute(1, 0, 2)  # [B, num_queries, C]
        bboxes = self.bbox_head(transformer_out)
        confidences = self.confidence_head(transformer_out)

        outputs = torch.cat([bboxes, confidences], dim=-1)  # [B, num_queries, 5]
        return outputs
