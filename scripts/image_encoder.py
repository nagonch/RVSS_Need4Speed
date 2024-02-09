import torch
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import Resize
import yaml

with open('scripts/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

class Regressor(torch.nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.squeezenet = squeezenet1_1(SqueezeNet1_1_Weights)
        # self.squeezenet.requires_grad = False
        self.linear = torch.nn.Linear(1000, 1)

    def forward(self, x):
        x = self.squeezenet(x)
        x = F.relu(x)
        x = self.linear(x)
        x = F.softmax(x)
        return x


if __name__ == "__main__":
    pass
