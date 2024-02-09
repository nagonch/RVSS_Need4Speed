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
import time
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

########## stop code paramaters
area_intrest = 30
min_size_thr = 10
max_size_thr = 50
## Blob detector setup
params = cv2.SimpleBlobDetector_Params()
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv2.SimpleBlobDetector(params)
else : 
    detector = cv2.SimpleBlobDetector_create(params)
def is_stop(im, detector, area_intrest, min_size_thr, max_size_thr):
    """Returns True if a red blob of big enough size is found in the image.
    Image is as a three dimensinal matrix (as if created by cv.read)"""
    im = np.array(im) 
    im = im[100:,:] #remove top
    imgh, imgw, _ = im.shape
    keypoints = detector.detect(im)
    for blob in keypoints:
        x, y = int(blob.pt[0]), int(blob.pt[1])
        size = blob.size
        x_min, y_min, x_max, y_max = max(x - area_intrest, 0), max(y - area_intrest, 0), min(x + area_intrest, imgw), min(y + area_intrest, imgh)
        im_near_blob = im[y_min:y_max,x_min:x_max,:]
        # is any of the area around it red?
        lower_red1 = np.array([0,0,0])
        upper_red1 = np.array([20,255,255])
        lower_red = np.array([90, 0, 150])    # Lower bound for red hue
        upper_red = np.array([179, 255, 255])    # Upper bound for red hue=
        mask1 = cv.inRange(im_near_blob, lower_red, upper_red)
        mask2 = cv.inRange(im_near_blob, lower_red1, upper_red1)
        mask = mask1+mask2
        result = cv.bitwise_and(im_near_blob, im_near_blob, mask=mask)
        if np.any(result != im_near_blob): 
            # check not black
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([50, 50, 50])
            pixel_color = im[y, x]
            if all(black_lower[i] <= pixel_color[i] <= black_upper[i] for i in range(3)):
                break
            if (min_size_thr < size < max_size_thr):
                # print('correct size')
                # cv.imshow("mask", result)
                # cv.waitKey(0)
                return True
        else:
            print("no stop sign")
        return False


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
stop_tracker = [0] #'0' if not a stop, '1' if a stop
i = 0
try:
    angle = 0
    while True:
        # get an image from the the robot
        bot_img = bot.getImage()
        img_shape = bot_img.shape
        img = bot_img[img_shape[0]//3:, :, :]
        img = torch.tensor(img).permute(-1, 0, 1)
        img = img_transform(img).float().cuda()
        img = img[None, :, :, :]
        pred = float(model(img)[0][0].detach().cpu().numpy())
        ############# STOP code
        stop = is_stop(bot_img, detector, area_intrest, min_size_thr, max_size_thr)
        if stop:
            stop_tracker.append(1)
        else:
            stop_tracker.append(0)
        # Is this the first time encountering this sign (in the last 5 images)
        max_len = min(len(stop_tracker),5)
        old_stop = False
        for j in range(max_len):
            if j > len(stop_tracker):
                break
            if stop_tracker[i - j] == 1:
                old_stop = True
        # If first time encountering image, then stop
        if not old_stop and stop:
            print("STOP")
            bot.setVelocity(0,0)
            time.sleep(0.5)
        ######### End of STOP code
        angle = pred
        print(angle)
        Kd = CONFIG['wheel-speed'] #base wheel speeds, increase to go faster, decrease to go slower
        Ka = CONFIG['turn-speed'] #how fast to turn when given an angle
        left  = int(Kd + Ka*angle)
        right = int(Kd - Ka*angle)
            
        bot.setVelocity(left, right)
        i += 1
            
        
except KeyboardInterrupt:    
    bot.setVelocity(0, 0)