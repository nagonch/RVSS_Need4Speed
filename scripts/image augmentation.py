import math
import numpy as np
import matplotlib
import machinevisiontoolbox 
import cv2

img = cv2.imread()#read image file 

img_v = cv2.flip(img, 1) # flip vertically 

# display image 
cv2.imshow("Vertical Flip", img_v)
cv2.waitKey(0)
cv2.destroyAllWindows()

#flip steering angle 

#rotate(frame, [angle])
