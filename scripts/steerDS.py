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
from PIL import Image

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext
        self.filenames = []
        for folder in os.listdir(root_folder):
            for item in os.listdir(f'{root_folder}/{folder}'):
                file = f'{root_folder}/{folder}/{item}'
                if file.endswith('.jpg') and not ")" in file and not "()" in file:
                    self.filenames.append(f'{root_folder}/{folder}/{item}')
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((224, 224), antialias=True)
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]
        with Image.open(f) as im:
            img = np.array(im)        
        img_shape = img.shape
        img = img[img_shape[0]//3:, :, :]
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(float(steering) >= 0) 
        img = self.resize(img)
        return img, steering

if __name__ == "__main__":
    pass
    # DS = SteerDataSet("data")
    # from matplotlib import pyplot as plt
    # for item in DS:
    #     img = item[0]
    #     avg_color = img.mean(axis=(1,2))
    #     plt.imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    #     plt.show()
    #     plt.close()
    #     print(avg_color.argmax())