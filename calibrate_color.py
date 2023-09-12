import cv2

import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
from image_processing.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import time
from utils import optimal_rotation_and_translation
import pickle
from image_processing.black_white import black_and_white_threshold
from image_processing.color_picker import Color_Picker


# Parameters
d = 0 
MIRROR_ROTATION_DEG = 45 # incidence angle of incoming laser ray (degree)
NUM_ITERATIONS = 50 # number captured images to detect laser position. Laser positions are averaged.
CALIBRATION_SAVE_PATH = "calibration_parameters" # calibration result save path
LOWER_RED = np.array([140,   10, 240])  # lower range of the laser color in HSV
UPPER_RED = np.array([180, 130, 256]) # upper range of the laser color in HSV
SAMPLE_X = 5 # number of calibration points in x-direction
SAMPLE_Y = 3 # number of calibration points in y-direction
TARGET_PLANE_DISTNCE = 560 # distance of target plane in mm

# Parameters






a = np.linspace(-150, 300, SAMPLE_X)
b = np.linspace(-150, 100, SAMPLE_Y)
x_t = np.tile(a, SAMPLE_Y)
y_t = np.repeat(b, SAMPLE_X)
z_t = [ 560 ]

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
    coordinate_transform = CoordinateTransform(d=d, D=z, rotation_degree=MIRROR_ROTATION_DEG)



    y_m, x_m = coordinate_transform.target_to_mirror(y_t, x_t) # order is changed in order to change x and y axis

    
    input(f"Press enter to start calibration in distance {z} from mirror.")
    # Set mirror position
    for i in range(len(x_m)):

        # Point the laser
        si_0.SetXY(y_m[i])        
        si_1.SetXY(x_m[i])    

        time.sleep(0.5)

        average_camera_coordinates_np = np.zeros((3), dtype=float)
        number_of_camera_coordimnates_in_batch = 0

        for  iter in range(NUM_ITERATIONS):
            # Get capture
            capture = device.update()

            # Get the color image from the capture
            ret_color, color_image = capture.get_color_image()

            # Get the colored depth
            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

            if not ret_color or not ret_depth:
                continue

            circle = detect_circle_position(color_image, lower_range=LOWER_RED, upper_range=UPPER_RED)

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
           

            # Show detected laser position
            cv2.imshow('Laser Detector', color_image)
            camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z])
            average_camera_coordinates_np += camera_coordinates
            number_of_camera_coordimnates_in_batch += 1

        laser_coordinates = [x_t[i], y_t[i], z]
        if number_of_camera_coordimnates_in_batch == 0:
            continue
        average_camera_coordinates_np /= number_of_camera_coordimnates_in_batch

        print("laser coordinates ", laser_coordinates)
        print("average camera coordinates ", average_camera_coordinates_np)

        laser_points.append(laser_coordinates)
        camera_points.append(average_camera_coordinates_np.tolist())

        cv2.waitKey(0)
    




if len(laser_points) < 3:
    print("Not enough points")
else:

    laser_points_np = np.array(laser_points).T
    camera_points_np = np.array(camera_points).T

    R, t = optimal_rotation_and_translation(camera_points_np, laser_points_np)

    print("Rotation matrix")
    print(R)

    print("translation vector")
    print(t)

    for i in range(len(laser_points_np[0])):
        print("Laser Point: {} , Camera Point: {}".format(laser_points_np[:, i], camera_points_np[:, i]))

    calibration_dict = {"R": R,
                        "t": t,
                        "laser_points": laser_points_np,
                        "camera_points": camera_points_np}

    with open('{}/parameters.pkl'.format(CALIBRATION_SAVE_PATH), 'wb') as f:
        pickle.dump(calibration_dict, f)

   
mre2.disconnect()
print("done")