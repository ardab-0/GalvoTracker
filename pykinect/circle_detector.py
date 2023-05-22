"""
https://github.com/ChristophRahn/red-circle-detection/blob/master/red-circle-detection.py

https://colorizer.org/
"""

import numpy as np
import cv2



def detect_circle_position(image, lower_range, upper_range):
    """
    finds circles in image with given color range in HSV color space and returns a 2d array with format [x_coordinate, y_coordinate, radius] 
    """

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    

    imgThreshHigh = cv2.inRange(hsv, lower_range, upper_range)
    thresh = imgThreshHigh.copy()
    
    captured_frame_hsv_red = cv2.GaussianBlur(thresh, (15, 15), 4, 4)

    captured_frame_hsv_red[np.abs(captured_frame_hsv_red) > 20] += 100
    # cv2.imshow('thresh', captured_frame_hsv_red)
    # Use the Hough transform to detect circles in the image
    circles = cv2.HoughCircles(captured_frame_hsv_red, cv2.HOUGH_GRADIENT, 2, captured_frame_hsv_red.shape[0] / 2, param1=100, param2=18, minRadius=3, maxRadius=60)

    if circles is not None:        
        return circles

    return None


def test_detect_circle_position():
    # define range of red color in HSV 
    lower_red = np.array([150,0,230]) 
    upper_red = np.array([170,100,255])

    im = cv2.imread("test_images/laser_im6.jpg")
    circles = detect_circle_position(im, lower_red, upper_red)
    print(circles)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        cv2.circle(im, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)

    # Display the resulting frame, quit with q
    cv2.imshow('frame', im)
    cv2.waitKey(0) 


if __name__ == "__main__":
    test_detect_circle_position()