import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t
import numpy as np
from image_processing.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import time
from utils import optimal_rotation_and_translation
import pickle
from image_processing.black_white import black_and_white_threshold
from image_processing.color_picker import Color_Picker


# Constants 
d = 0
mirror_rotation_deg = 45
num_iterations = 50
save_path = "ir_calibration_parameters"


lower_red = np.array([140,   10, 240]) 
upper_red = np.array([180, 130, 256])

with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]


sample_x = 5
sample_y = 3
a = np.linspace(0, 100, sample_x)
b = np.linspace(0, 50, sample_y)

x_t = np.tile(a, sample_y)
y_t = np.repeat(b, sample_x)
z_t = [ 510 ]

laser_points = []
camera_points = []


# Initialize the pykinect library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_YUY2
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)


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




for z in z_t:
    coordinate_transform = CoordinateTransform(d=d, D=z, rotation_degree=mirror_rotation_deg)



    y_m, x_m = coordinate_transform.target_to_mirror(y_t, x_t) # order is changed in order to change x and y axis

    
    input(f"Press enter to start calibration in distance {z} mm from mirror.")
    # Set mirror position
    for i in range(len(x_m)):

        # Point the laser
        si_0.SetXY(y_m[i])        
        si_1.SetXY(x_m[i])    

        time.sleep(0.5)

        reference_point_in_laser_coordinates = np.array([x_t[i], y_t[i], z]).reshape((3, 1))
        reference_point_in_camera_coordinates = R.T @ (reference_point_in_laser_coordinates - t)
        reference_point_in_camera_coordinates_k4a = k4a_float3_t((reference_point_in_camera_coordinates[0], reference_point_in_camera_coordinates[1], reference_point_in_camera_coordinates[2]))


        reference_pos2d_color = device.calibration.convert_3d_to_2d(reference_point_in_camera_coordinates_k4a, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        reference_pos2d_color = [reference_pos2d_color.xy.x, reference_pos2d_color.xy.y]

        average_laser_pos_in_camera_coordinates_np = np.zeros((3), dtype=float)
        number_of_camera_coordinates_in_batch = 0
        while  number_of_camera_coordinates_in_batch < num_iterations:
            # Get capture
            capture = device.update()

            # Get the color image from the capture
            ret_color, color_image = capture.get_color_image()

            # Get the colored depth
            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

            if not ret_color or not ret_depth:
                continue

            #thresholded_image = black_and_white_threshold(color_image)    
            circle = detect_circle_position(color_image, lower_range=lower_red, upper_range=upper_red)

            if circle is  None:
                print("Laser is not detected")                
                continue

            circle = np.round(circle).astype("int")
            cv2.circle(color_image, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)

            pix_x = circle[0]
            pix_y = circle[1]
            rgb_depth = transformed_depth_image[pix_y, pix_x]

            pixels = k4a_float2_t((pix_x, pix_y))

            pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
            # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
            # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")
        
            cv2.circle(color_image, center=(int(reference_pos2d_color[0]), int(reference_pos2d_color[1])), radius=circle[2], color=(0, 0, 255), thickness=2)   
            cv2.imshow('Laser Detector', color_image)
            laser_in_camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z])
            average_laser_pos_in_camera_coordinates_np += laser_in_camera_coordinates
            number_of_camera_coordinates_in_batch += 1

        


        if number_of_camera_coordinates_in_batch == 0:
            continue
        average_laser_pos_in_camera_coordinates_np /= number_of_camera_coordinates_in_batch

        average_laser_pos_in_laser_coordinates_np = R @ average_laser_pos_in_camera_coordinates_np.reshape((3, 1)) + t

        rmse = np.sqrt(np.mean(np.square(average_laser_pos_in_laser_coordinates_np - reference_point_in_laser_coordinates)))
        print("RMSE: ", rmse)


        cv2.waitKey(0)
    



   
mre2.disconnect()
print("done")