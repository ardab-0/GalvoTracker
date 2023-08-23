import cv2
import serial
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
from utils import optimal_rotation_and_translation

# Constants 

d = 0
mirror_rotation_deg = 45
save_path = "ir_calibration_parameters_test"
distance_of_sensor_from_marker_mm=-30
PI_COM_PORT = "COM6"
CALIBRATION_ITER = 10



with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]
    laser_points = loaded_dict["laser_points"]
    camera_points = loaded_dict["camera_points"]


laser_points = laser_points[:, :3*CALIBRATION_ITER]
camera_points = camera_points[:, :3*CALIBRATION_ITER]
R, t = optimal_rotation_and_translation(camera_points, laser_points)

def get_fine_laser_positions(rough_laser_coords, search_length_mm=10, delta_mm=0.4):
    """
        rough_laser_coords: list
    """

    fine_coords_3d = []
    for i, coord in enumerate(rough_laser_coords):
        id = i+1 # sensor ids start from 1
        sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=coord, width_mm=search_length_mm, height_mm=search_length_mm, delta_mm=delta_mm, sensor_id=id)
        fine_coords_3d.append(max_pos)

    # plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    # plt.show()

    return fine_coords_3d


def search_for_laser_position(initial_position_mm, width_mm, height_mm, delta_mm, sensor_id=1):
    """
        initial_position_mm: length 3 list
        width_mm: search area width
        height_mm: search area height
        delta_mm: search step size
    """
    sample_x = int(width_mm / delta_mm) + 1
    sample_y = int(height_mm / delta_mm) + 1
    w = np.linspace(-width_mm / 2, width_mm / 2, sample_x)
    h = np.linspace(-height_mm / 2, height_mm / 2, sample_y)
    x_t = np.tile(w, sample_y)
    y_t = np.repeat(h, sample_x)

    x_t += initial_position_mm[0]
    y_t += initial_position_mm[1]
    z_t = initial_position_mm[2]

    coordinate_transform = CoordinateTransform(
        d=d, D=z_t, rotation_degree=mirror_rotation_deg
    )
    y_m, x_m = coordinate_transform.target_to_mirror(
        y_t, x_t
    )  # order is changed in order to change x and y axis

    sensor_readings = []

    for i in range(len(x_m)):
        # Point the laser
        si_0.SetXY(y_m[i])
        si_1.SetXY(x_m[i])

        time.sleep(0.001)
        sensor_readings.append(get_sensor_reading(sensor_id))

    sensor_readings = np.array(sensor_readings)
    max_idx = np.argmax(sensor_readings)

    sensor_data = np.reshape(sensor_readings, (sample_y, sample_x))
    coordinate_axes = (w+initial_position_mm[0], h+initial_position_mm[1])
    max_position_mm = (x_t[max_idx], y_t[max_idx], z_t)
    target_coordinates = (x_t, y_t, z_t) # z_t is not ndarray


    return sensor_data, coordinate_axes, max_position_mm, target_coordinates

def get_sensor_reading(sensor_id): 
    """
    sensor_id: int
    """   
    s.flush()
    s.write(f"pd_{sensor_id}\n".encode())
    mes = s.read_until()
    return float(mes.decode().strip("\r\n"))


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
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
device_config.synchronized_images_only = False
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)

cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
font = cv2.FONT_HERSHEY_SIMPLEX

# initilaize serial port to PICO
s = serial.Serial(port=PI_COM_PORT, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1)

def main():
    # gives undefined warning but works (pybind11 c++ module) change import *
    prevCircle = CircleClass()
    circle_detector = CircleDetectorClass(1920, 1080) # K4A_COLOR_RESOLUTION_1080P

    prev_3d_coor = 0
    speed_timer = 1
    while True:
        capture = device.update()
        ret_color, color_image = capture.get_color_image()

        if not ret_color:
            continue

        new_circle = circle_detector.detect_np(color_image, prevCircle)    
        prevCircle = new_circle
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()        
        if not ret_depth:
            continue  
        
        pix_x = int(new_circle.x)
        pix_y = int(new_circle.y)
        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)

        target_in_camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))


        # point to target 
        target_in_camera_coordinates += np.array([0,distance_of_sensor_from_marker_mm ,0]).reshape((3, 1))

        # rotate and translate  
        target_in_laser_coordinates =  R @ target_in_camera_coordinates + t
    

        coordinate_transform = CoordinateTransform(d=d, D=target_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)
        y_m, x_m = coordinate_transform.target_to_mirror(target_in_laser_coordinates[1], target_in_laser_coordinates[0]) # order is changed in order to change x and y axis

        
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0])        


        color_image = cv2.circle(color_image, (int(new_circle.x), int(new_circle.y)), radius=10, color=(0, 255, 0), thickness=2)


            
        # Show detected target position
        cv2.imshow('Laser Detector',color_image)
        # Press q key to stop

        if cv2.waitKey(1) == ord("m"):
            sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=target_in_laser_coordinates.reshape((-1)), width_mm=10, height_mm=10, delta_mm=0.2, sensor_id=2)
            max_pos = np.array(max_pos).reshape(3,1)
            rmse = np.sqrt(np.mean(np.square(max_pos - target_in_laser_coordinates)))
            print(rmse)
            cv2.waitKey(0)



        if cv2.waitKey(1) == ord('q'):
            break

        
    mre2.disconnect()
    print("done")


if __name__ == "__main__":
    main()
