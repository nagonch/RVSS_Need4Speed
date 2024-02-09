#!/usr/bin/env python
import time
import sys
import os
import cv2
import numpy as np
from pynput import keyboard
import yaml

with open('scripts/collect_config.yaml', 'r') as file:
    CONFIG = yaml.safe_load(file)
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))


FOLDER_NAME = CONFIG['folder']
if not os.path.isdir(f'data/{FOLDER_NAME}'):
    os.makedirs(f'data/{FOLDER_NAME}')
if not os.path.exists(script_path+"/../data/"+CONFIG['folder']):
    data_path = script_path.replace('scripts', 'data')
    print(f'Folder "{FOLDER_NAME}" in path {data_path} does not exist. Please create it.')
    exit()

from pibot_client import PiBot
bot = PiBot(ip=CONFIG['ip-num'])
# stop the robot

bot.setVelocity(0, 0)

#countdown before beginning
print("Get ready...")
print("GO!")


# Initialize variables
angle = 0
im_number = CONFIG['im-num']
continue_running = True
print(im_number)

def on_press(key):
    global angle, continue_running, im_number
    try:
        if key == keyboard.Key.up:
            print('takeing imag')
            img = bot.getImage()
            cv2.imwrite(script_path+"/../data/"+CONFIG['folder']+"/"+str(im_number).zfill(6)+'%.2f'%angle+".jpg", img) 
            im_number += 1

        elif key == keyboard.Key.space:
            print("stop")
            bot.setVelocity(0, 0)
            continue_running = False
            # return False  # Stop listener

    except Exception as e:
        print(f"An error occurred: {e}")
        bot.setVelocity(0, 0)

# Start the listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while continue_running:
        # Get an image from the robot
        
        time.sleep(0.1)  # Small delay to reduce CPU usage

    # Clean up
    bot.setVelocity(0, 0)
    listener.stop()
    print("Script ended")


except KeyboardInterrupt:    
    bot.setVelocity(0, 0)
    listener.stop()

