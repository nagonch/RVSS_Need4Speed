import torch
from steerDS import SteerDataSet
from image_encoder import Regressor
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import yaml
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from torcheval.metrics.functional import binary_f1_score
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

with open('scripts/train_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

def get_data(dataset_path):
    dataset = SteerDataSet(dataset_path)
    train_size = CONFIG['train-size']
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, 1-train_size])
    train_dataloader = DataLoader(train_set, batch_size=CONFIG['batch-size'], shuffle=True)
    test_dataloader = DataLoader(val_set, batch_size=CONFIG['batch-size'], shuffle=True)

    return train_dataloader, test_dataloader

def weighted_mse_loss(input, target, weight):
    loss = F.cross_entropy(input, target, reduction='none') * weight
    return torch.sum(loss)

def train(model, train_dataloader, test_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
    loss = nn.MSELoss()
    loss_values = []
    val_losses = []
    val_metrics = []
    for epoch in tqdm(range(CONFIG['n-epochs'])):
        loss_epoch = []
        for item in train_dataloader:
            optimizer.zero_grad()
            x, y = item
            x = x.cuda()
            y = y.cuda()
            y_pred = model(x).reshape(-1)
            weights = x.mean(axis=(2, 3)).argmax()
            weights = (weights == 2).to(torch.float32) * CONFIG['orange-coeff'] + (weights != 2).to(torch.float32)
            #loss_train = weighted_mse_loss(y_pred, y, weights.cuda())
            loss_train = L1loss(y_pred, y, weights.cuda()) # L1loss suggested by Dimity
            loss_epoch.append(loss_train)
            loss_train.backward()
            optimizer.step()
            scheduler.step(epoch)
        epoch_loss = torch.mean(torch.tensor(loss_epoch))
        print(f"{epoch}: {epoch_loss}")
        loss_values.append(epoch_loss)
        torch.save(model.state_dict(), f'training/{CONFIG["model-name"]}.pt')
        plt.plot(loss_values)
        plt.show()
        plt.savefig(f'training/{CONFIG["model-name"]}_loss_train.png')
        plt.close()
    
        with torch.no_grad():
            val_loss = []
            val_metric = []
            for item in test_dataloader:
                x, y = item
                x = x.cuda()
                y = y.cuda()
                y_pred = model(x).reshape(-1)
                loss_val = loss(y_pred, y)
                val_loss.append(loss_val.cpu())
                metric = binary_f1_score(y_pred, y, threshold=0.5)
                val_metric.append(metric)
            val_losses.append(torch.tensor(val_loss).mean())
            val_metrics.append(torch.tensor(val_metric).mean())
        plt.plot(val_losses)
        plt.show()
        plt.savefig(f'training/{CONFIG["model-name"]}_loss_val.png')
        plt.close()
        plt.plot(val_metrics)
        plt.show()
        plt.savefig(f'training/{CONFIG["model-name"]}_metric_val.png')
        plt.close()

if __name__ == "__main__":
    train_dataloader, test_dataloader = get_data("data")
    model = Regressor().cuda()
    if CONFIG['weights'] != "none":
        model_state = torch.load(CONFIG['weights'])
        model.load_state_dict(model_state)
    result = train(model, train_dataloader, test_dataloader)
