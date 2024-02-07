import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize
import yaml

with open('scripts/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

class ViTRegressor(torch.nn.Module):
    def __init__(self):
        super(ViTRegressor, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.linear = torch.nn.Linear(1000, 1)

    def forward(self, x):
        x = self.vit(x)
        x = F.relu(x)
        x = self.linear(x)
        x = F.tanh(x) * CONFIG['max-abs-angle']
        return x


if __name__ == "__main__":
    pass
