import torch
from steerDS import SteerDataSet
from image_encoder import ViTRegressor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import yaml
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

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
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = StepLR(optimizer, step_size=CONFIG['sched-step'], gamma=CONFIG['sched-gamma'])
    loss = nn.MSELoss()
    loss_values = []
    for epoch in tqdm(range(CONFIG['n-epochs'])):
        loss_epoch = []
        for item in train_dataloader:
            optimizer.zero_grad()
            x, y = item
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x).reshape(-1)
            loss_val = loss(y_pred, y)
            loss_epoch.append(loss_val)
            loss_val.backward()
            optimizer.step()
            scheduler.step()
        epoch_loss = torch.mean(torch.tensor(loss_epoch))
        print(f"{epoch}: {epoch_loss}")
        loss_values.append(epoch_loss)
        plt.plot(loss_values)
        plt.show()
        plt.savefig('loss.png')
        plt.close()

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_data("data/track3")
    model = ViTRegressor().cuda()
    result = train(model, train_dataloader)
