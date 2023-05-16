"""
https://github.com/ChristophRahn/red-circle-detection/blob/master/red-circle-detection.py
"""

import numpy as np
import cv2



im = cv2.imread("test_images/laser_im.jpg")

# Convert BGR to HSV

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV 
lower_red = np.array([160,20,150]) 
upper_red = np.array([180,255,255])

imgThreshHigh = cv2.inRange(hsv, lower_red, upper_red)
thresh = imgThreshHigh.copy()

captured_frame_hsv_red = cv2.GaussianBlur(thresh, (5, 5), 2, 2)
# Use the Hough transform to detect circles in the image
circles = cv2.HoughCircles(captured_frame_hsv_red, cv2.HOUGH_GRADIENT, 1, captured_frame_hsv_red.shape[0] / 8, param1=100, param2=18, minRadius=5, maxRadius=60)

# If we have extracted a circle, draw an outline
# We only need to detect one circle here, since there will only be one reference object
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    cv2.circle(im, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)

# Display the resulting frame, quit with q
cv2.imshow('frame', im)



cv2.imshow('frame',im)
cv2.imshow('Object',thresh)
cv2.waitKey(0) 