import torch
from steerDS import SteerDataSet
from image_encoder import ViTRegressor
from torchvision.transforms import Resize

if __name__ == "__main__":
    ds = SteerDataSet("data/track3")
    for item in ds:
        transform = Resize((224, 224))
        img = transform(item[0]).cuda()
        break
    print(img.shape, img[None].shape)
    model = ViTRegressor().cuda()
    print(model(img[None]))
