import torch
from steerDS import SteerDataSet
from image_encoder import ViTRegressor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import yaml

with open('scripts/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

if __name__ == "__main__":
    dataset = SteerDataSet("data/track3")
    dataset_size = len(dataset)
    train_size = CONFIG['train-size']
    train_elements = int(dataset_size * train_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_elements, dataset_size-train_elements])
    train_dataloader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=16, shuffle=True)
    model = ViTRegressor().cuda()
    for item in train_dataloader:
        x, y = item
        x = x.cuda()
        y = y.cuda()
        print(model(x))
        raise
