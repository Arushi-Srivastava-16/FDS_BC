# breast_cancer_detection/models/backbone.py

import torch.nn as nn
import torchvision.models as models

class Backbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=True):
        super(Backbone, self).__init__()
        self.name = name.lower()

        if self.name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # remove final FC
            self.out_dim = 2048

        elif self.name == "swin":
            from timm import create_model
            model = create_model('swin_base_patch4_window7_224', pretrained=pretrained)
            self.feature_extractor = model.forward_features
            self.out_dim = 1024

        elif self.name == "efficientvit":
            from efficientvit.models.efficientvit import efficientvit_b2
            model = efficientvit_b2(pretrained=pretrained)
            self.feature_extractor = model.forward_features
            self.out_dim = 640

        else:
            raise ValueError(f"Backbone {name} not supported!")

    def forward(self, x):
        return self.feature_extractor(x)
