import torch
from steerDS import SteerDataSet
from image_encoder import ViTRegressor
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
    print(len(train_set), len(val_set))
    raise
    for item in ds:
        transform = Resize((224, 224))
        img = transform(item[0]).cuda()
        break
    print(img.shape, img[None].shape)
    model = ViTRegressor().cuda()
    print(model(img[None]))
