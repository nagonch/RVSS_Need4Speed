import numpy as np
import cv2
import os
import glob

#REPLACE with image path
path = "C:/Users/33jav/RVSS_Need4Speed/data/test_stop"

img_list = sorted(glob.glob(os.path.join(path, '*.jpg')))
i = 0

for img_path in img_list:
    img = cv2.imread(img_path)#read image file 
    img_v = cv2.flip(img, 1) # flip vertically 
    #flip angle
    base_name = os.path.basename(img_path)
    angle = float(os.path.splitext(str(base_name))[0][6:])
    new_angle = angle*-1
    new_name = str(os.path.splitext(str(base_name))[0][:6]+str(new_angle))
    cv2.imwrite(os.path.dirname(img_path)+"/"+new_name, img_v)
    i += 1