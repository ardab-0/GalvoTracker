import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
from pykinect.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import pickle
import time


# Constants 
# target color range
lower_range = np.array([50,100,30]) 
upper_range = np.array([100,230,150])


d = 0
mirror_rotation_deg = 45

save_path = "calibration_parameters"

y_offset = 20

with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]

# R = np.array([[ 0.99950191, -0.02218147,  0.02244816],
#                     [ 0.02269092,  0.99948479, -0.02269971],
#                     [-0.02193308,  0.02319778,  0.99949027]])
    
# t = np.array([[42.32738624],
#                 [42.51038257],
#                 [42.38330588]])


# initialize mirrors
mre2 = optoMDC.connect()
mre2.reset()

# Set up mirror in closed loop control mode(XY)
ch_0 = mre2.Mirror.Channel_0
ch_0.StaticInput.SetAsInput()                       # (1) here we tell the Manager that we will use a static input
ch_0.SetControlMode(optoMDC.Units.XY)           
ch_0.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_0 = mre2.Mirror.Channel_0.StaticInput


ch_1 = mre2.Mirror.Channel_1
ch_1.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
ch_1.SetControlMode(optoMDC.Units.XY)           
ch_1.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_1 = mre2.Mirror.Channel_1.StaticInput


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
    start = time.time()
    # Get capture
    capture = device.update()

    # Get the color image from the capture
    ret_color, color_image = capture.get_color_image()

    # Get the colored depth
    ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

    if not ret_color or not ret_depth:
        continue  

    circles = detect_circle_position(color_image, lower_range=lower_range, upper_range=upper_range)

    if circles is  None:
        print("Target is not detected")
        continue

    circles = np.round(circles[0, :]).astype("int")
    cv2.circle(color_image, center=(circles[0, 0], circles[0, 1]), radius=circles[0, 2], color=(0, 255, 0), thickness=2)
    # Show detected target position
    cv2.imshow('Laser Detector',color_image)


    pix_x = circles[0, 0]
    pix_y = circles[0, 1]
    rgb_depth = transformed_depth_image[pix_y, pix_x]

    pixels = k4a_float2_t((pix_x, pix_y))

    pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
    # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

    camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))

    # rotate and translate

    
    

    # camera_coordinates_in_laser_coordinates = camera_coordinates + np.array([130, 50, 70]).reshape((3, 1))  # should find parameters automatically

    camera_coordinates_in_laser_coordinates =  R @ camera_coordinates + t

    print("camera_coordinates", camera_coordinates)

    print("camera_coordinates_in_laser_coordinates", camera_coordinates_in_laser_coordinates)



    coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)



    y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1]+y_offset, camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis

    if(len(y_m) > 0 and len(x_m) > 0):
        si_0.SetXY(y_m[0])        
        si_1.SetXY(x_m[0]) 


    print("fps: ", 1 / (time.time() - start))
    # Press q key to stop
    if cv2.waitKey(1) == ord('q'):
        break


mre2.disconnect()
print("done")