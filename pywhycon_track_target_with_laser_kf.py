import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
from image_processing.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import pickle
import time
import os
from circle_detector_library.circle_detector_module import *
from kalman_filter.track_3d import SecondOrderKF, FirstOrderKF


# Constants 

d = 0
mirror_rotation_deg = 45
save_path = "calibration_parameters"

# filter coefficients 
R_std = 0.1
Q_std = 20
P_std = 100
next_t = 0.085 # seconds


with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]



def main():
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
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_YUY2
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    # Initialize kalman filter 
    tracker = SecondOrderKF(R_std=R_std, Q_std=Q_std, P_std=P_std)

    cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX


    # gives undefined warning but works (pybind11 c++ module) change import *
    prevCircle = CircleClass()
    circle_detector = CircleDetectorClass(1280, 720) # K4A_COLOR_RESOLUTION_720P
    start = 0
    while True:
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()
        
        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
        
        
        if not ret_color or not ret_depth:
            continue  
        
        
        #color_image_3channel = color_image[:, :, :3]
        # returns 0, 0 if target is not detected
        
        new_circle = circle_detector.detect_np(color_image, prevCircle)    
        prevCircle = new_circle

    
        
        pix_x = int(new_circle.x)
        pix_y = int(new_circle.y)
        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
        # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))

        # rotate and translate  
        camera_coordinates_in_laser_coordinates =  R @ camera_coordinates + t

        now = time.time()
        dt = now - start
        start = now
       
        x, P = tracker.update(dt, camera_coordinates_in_laser_coordinates.reshape(-1))
        x_pred = tracker.predict_position(x, next_t)


        #second order
        predicted_coordinates = np.array([x_pred[0, 0], x_pred[3, 0], x_pred[6, 0]]).reshape((3, 1))

        #first order
        #predicted_coordinates = np.array([x_pred[0, 0], x_pred[2, 0], x_pred[4, 0]]).reshape((3, 1))

        coordinate_transform = CoordinateTransform(d=d, D=predicted_coordinates[2], rotation_degree=mirror_rotation_deg)



        y_m, x_m = coordinate_transform.target_to_mirror(predicted_coordinates[1], predicted_coordinates[0]) # order is changed in order to change x and y axis

        
        
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0]) 


        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_image, f"fps: {1 / dt}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(color_image, f"Target Coordinates w.r.t. mirror center:", (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"X: {camera_coordinates_in_laser_coordinates[0]}", (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Y: {camera_coordinates_in_laser_coordinates[1]}", (10, 80), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Z: {camera_coordinates_in_laser_coordinates[2]}", (10, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.putText(color_image, f"X: {predicted_coordinates[0]}", (10, 120), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Y: {predicted_coordinates[1]}", (10, 140), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Z: {predicted_coordinates[2]}", (10, 160), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        
        # cv2.circle(color_image, center=(mouse_x, mouse_y), radius=10, color=(0, 255, 0), thickness=2)
        # Show detected target position
        cv2.imshow('Laser Detector',color_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

        

    mre2.disconnect()
    print("done")


if __name__ == "__main__":
    main()
