#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot
from torchvision.transforms import Resize
import yaml
from image_encoder import Regressor

with open('scripts/deploy_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)

bot = PiBot(ip=CONFIG['ip'])

# stop the robot 
bot.setVelocity(0, 0)

model = Regressor()
model_state = torch.load(CONFIG['model-path'])
model.load_state_dict(model_state)
model = model.cuda()

#countdown before beginning
print("Get ready...")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("GO!")

img_transform = Resize((224, 224), antialias=True)

try:
    angle = 0
    while True:
        # get an image from the the robot
        img = bot.getImage()
        img_shape = img.shape
        img = img[img_shape[0]//3:, :, :]
        img = torch.tensor(img).permute(-1, 0, 1)
        img = img_transform(img).float().cuda()
        img = img[None, :, :, :]
        pred = float(model(img)[0][0].detach().cpu().numpy())
        #TODO: stop??
        angle = pred
        print(angle)
        Kd = CONFIG['wheel-speed'] #base wheel speeds, increase to go faster, decrease to go slower
        Ka = CONFIG['turn-speed'] #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)