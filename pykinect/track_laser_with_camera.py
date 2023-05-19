import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
from circle_detector import detect_circle_position



# Initialize the library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)

cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
while True:
    
    # Get capture
    capture = device.update()

    # Get the color image from the capture
    ret_color, color_image = capture.get_color_image()

    # Get the colored depth
    ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    if not ret_color or not ret_depth:
        continue
    

    lower_red = np.array([150,0,230]) 
    upper_red = np.array([170,100,255])



    circles = detect_circle_position(color_image, lower_range=lower_red, upper_range=upper_red)

    if circles is  None:
        print("Laser is not detected")
        continue

    circles = np.round(circles[0, :]).astype("int")
    cv2.circle(color_image, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)

    pix_x = circles[0, 0]
    pix_y = circles[0, 1]
    rgb_depth = transformed_depth_image[pix_y, pix_x]

    pixels = k4a_float2_t((pix_x, pix_y))

    pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
    print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

    # Show detected laser position
    cv2.imshow('Laser Detector',color_image)

    # Press q key to stop
    if cv2.waitKey(1) == ord('q'):
        break
