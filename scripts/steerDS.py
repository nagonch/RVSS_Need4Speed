import numpy as np
from glob import glob
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset
import cv2
from glob import glob
from os import path
import os
import torch

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext
        self.filenames = []
        for folder in os.listdir(root_folder):
            for item in os.listdir(f'{root_folder}/{folder}'):
                file = f'{root_folder}/{folder}/{item}'
                if file.endswith('.jpg'):
                    self.filenames.append(f'{root_folder}/{folder}/{item}')
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.colorjit = v2.ColorJitter()
        self.flip = v2.RandomHorizontalFlip(p=0.5)
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)
        img_shape = img.shape
        img = img[img_shape[0]//3:, :, :]
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(float(steering)) 
        img = self.resize(img)
        return img, steering

if __name__ == "__main__":
    pass
    DS = SteerDataSet("data")
    for item in DS:
        print(item[0].shape, item[1])
        raise