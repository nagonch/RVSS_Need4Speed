import cv2 as cv
import numpy as np
import os
import glob

#############  Threashold Parameters
path = "C:/Users/33jav/RVSS_Need4Speed/data/test_stop"
#path = "C:/Users/33jav/RVSS_Need4Speed/data/long_test2"
i = 0
area_intrest = 30
min_size_thr = 10
max_size_thr = 50

## Blob detector setup
params = cv.SimpleBlobDetector_Params()
ver = (cv.__version__).split('.')
if int(ver[0]) < 3 :
    detector = cv.SimpleBlobDetector(params)
else : 
    detector = cv.SimpleBlobDetector_create(params)

imglist = sorted(glob.glob(os.path.join(path, '*.jpg')))


### function to detect blobs
def is_stop(im, detector, area_intrest, min_size_thr, max_size_thr):
    """Returns True if a red blob of big enough size is found in the image.
    Image is as a three dimensinal matrix (as if created by cv.read)"""
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
        # cv.imshow("zoomed", im_near_blob)
        # cv.waitKey(0)
    # im_with_keypoints = cv.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow("Keypoints", im_with_keypoints)
    # cv.waitKey(0)

for img in imglist:
    if i > 100:
        break
    i += 1
    print(f'img = {os.path.basename(img)}')
    im = cv.imread(img)
    stop = is_stop(im, detector, area_intrest, min_size_thr, max_size_thr)
    if stop:
        print("stop detected")
    




