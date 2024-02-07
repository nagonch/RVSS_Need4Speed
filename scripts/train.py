import torch
from steerDS import SteerDataSet
from image_encoder import ViTRegressor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import yaml
from tqdm import tqdm

with open('scripts/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

def get_data(dataset_path):
    dataset = SteerDataSet(dataset_path)
    dataset_size = len(dataset)
    train_size = CONFIG['train-size']
    train_elements = int(dataset_size * train_size)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_elements, dataset_size-train_elements])
    train_dataloader = DataLoader(train_set, batch_size=CONFIG['batch-size'], shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=CONFIG['batch-size'], shuffle=True)

    return train_dataloader, test_dataloader

def train(model, train_dataloader):
    for epoch in tqdm(range(CONFIG['n-epochs'])):
        for item in train_dataloader:
            x, y = item
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x)
            y_pred = torch.clip(y_pred, min=-CONFIG['max-abs-angle'], max=CONFIG['max-abs-angle'])
            print(y_pred)

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_data("data/track3")
    model = ViTRegressor().cuda()
    result = train(model, train_dataloader)
