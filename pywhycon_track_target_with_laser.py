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


# Parameters 
d = 0
MIRROR_ROTATION_DEG = 45 # incidence angle of incoming laser ray (degree)
CALIBRATION_SAVE_PATH = "calibration_parameters"
CAPTURE_VIDEO = False
# Parameters 


with open('{}/parameters.pkl'.format(CALIBRATION_SAVE_PATH), 'rb') as f:
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
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
    device_config.synchronized_images_only = False
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
    font = cv2.FONT_HERSHEY_SIMPLEX


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1280,720))

    # gives undefined warning but works (pybind11 c++ module) change import *
    prevCircle = CircleClass()
    circle_detector = CircleDetectorClass(1280, 720) # K4A_COLOR_RESOLUTION_720P

    prev_3d_coor = 0
    speed_timer = 1
    while True:
        start = time.time()
        # Get capture
        capture = device.update()
        print("Time until capture (s): ", time.time() - start)

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()
        print("Time until color image (s): ", time.time() - start)

        if not ret_color:
            continue

        new_circle = circle_detector.detect_np(color_image, prevCircle)    
        prevCircle = new_circle
        print("Time until circle detection (s): ", time.time() - start)

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
        print("Time until depth image (s): ", time.time() - start)

        
        if not ret_depth:
            continue  
        
        
        #color_image_3channel = color_image[:, :, :3]
        # returns 0, 0 if target is not detected
        
       

    
        
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
    

        coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2].item(), rotation_degree=MIRROR_ROTATION_DEG)
        y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1], camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis
        print("Time until completing calculations (s): ", time.time() - start)

        
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0])        


       
        dt = time.time() - speed_timer
        speed_timer = time.time()

        speed = (camera_coordinates_in_laser_coordinates - prev_3d_coor) / dt
        prev_3d_coor = camera_coordinates_in_laser_coordinates
        

        end = time.time()
        print("elapsed time: ", (end - start))    
        cv2.putText(color_image, f"fps: {1 / (end - start)}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Target Coordinates w.r.t. mirror center:", (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"X (mm): {camera_coordinates_in_laser_coordinates[0]}", (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Y (mm): {camera_coordinates_in_laser_coordinates[1]}", (10, 80), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(color_image, f"Z (mm): {camera_coordinates_in_laser_coordinates[2]}", (10, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # cv2.putText(color_image, f"VX (mm/s): {speed[0]}", (10, 120), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"VY (mm/s): {speed[1]}", (10, 140), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"VZ (mm/s): {speed[2]}", (10, 160), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        color_image = cv2.circle(color_image, (int(new_circle.x), int(new_circle.y)), radius=10, color=(0, 255, 0), thickness=2)

        if CAPTURE_VIDEO:
            out.write(color_image)
            
        # Show detected target position
        cv2.imshow('Laser Detector',color_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

        
    out.release()
    mre2.disconnect()
    print("done")


if __name__ == "__main__":
    main()
