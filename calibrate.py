import cv2
import serial
import pykinect_azure as pykinect
from pykinect_azure import (
    K4A_CALIBRATION_TYPE_COLOR,
    K4A_CALIBRATION_TYPE_DEPTH,
    k4a_float2_t,
)
import numpy as np
from circle_detector_library.circle_detector_module import *
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import time
from utils import optimal_rotation_and_translation, argsort
import pickle
import sys
import matplotlib.pyplot as plt
from image_processing.local_maxima_finding import find_local_maxima
import tkinter as tk


# Parameters
d = 0  # distance between mirror surface and rotation center
MIRROR_ROTATION_DEG = 45 # incidence angle of incoming laser ray (degree)
CALIBRATION_SAVE_PATH = "calibration_parameters"  # calibration result save path
CAPTURE_COUNT = 5 # number of chessboard images captured for averaging
ITER_COUNT = 10 # number of calibration iterations / number of calibration positions
PI_COM_PORT = "COM6" # COM port used by raspberry pi pico
SENSOR_POS_WRT_MARKER = -55 # location of middle sensor with respect to center of chessboard calibration pattern (mm)
SENSOR_DISTANCE = 75 # distance between sensors (mm)
# Parameters


# global variables
mirror_x = 0
mirror_y = 0
mirror_z = 0

# Initialize the pykinect library, if the library is not found, add the library path as argument
pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
# print(device_config)

# Start device
device = pykinect.start_device(config=device_config)

# initialize mirrors
mre2 = optoMDC.connect()
mre2.reset()

# Set up mirror in closed loop control mode(XY)
ch_0 = mre2.Mirror.Channel_0
ch_0.StaticInput.SetAsInput()  # (1) here we tell the Manager that we will use a static input
ch_0.SetControlMode(optoMDC.Units.XY)
ch_0.Manager.CheckSignalFlow()  # This is a useful method to make sure the signal flow is configured correctly.
si_0 = mre2.Mirror.Channel_0.StaticInput

ch_1 = mre2.Mirror.Channel_1

ch_1.StaticInput.SetAsInput()  # (1) here we tell the Manager that we will use a static input
ch_1.SetControlMode(optoMDC.Units.XY)
ch_1.Manager.CheckSignalFlow()  # This is a useful method to make sure the signal flow is configured correctly.
si_1 = mre2.Mirror.Channel_1.StaticInput


# initilaize serial port to PICO
s = serial.Serial(port=PI_COM_PORT, parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1)



def get_sensor_reading(sensor_id): 
    """
    sensor_id: int
    """   
    s.flush()
    s.write(f"pd_{sensor_id}\n".encode())
    mes = s.read_until()
    return float(mes.decode().strip("\r\n"))


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
        d=d, D=z_t, rotation_degree=MIRROR_ROTATION_DEG
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



def search_for_multiple_laser_position(initial_position_mm, width_mm, height_mm, delta_mm, sensor_ids):
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
        d=d, D=z_t, rotation_degree=MIRROR_ROTATION_DEG
    )
    y_m, x_m = coordinate_transform.target_to_mirror(
        y_t, x_t
    )  # order is changed in order to change x and y axis

    multiple_sensor_readings = []
    
       
    for i in range(len(x_m)):
        sensor_readings = []
        for id in sensor_ids:
            # Point the laser
            si_0.SetXY(y_m[i])
            si_1.SetXY(x_m[i])

            time.sleep(0.001)
            sensor_readings.append(get_sensor_reading(id))

        multiple_sensor_readings.append(sensor_readings)


    multiple_sensor_readings = np.array(multiple_sensor_readings)
    max_indices = np.argmax(multiple_sensor_readings, axis=0)

    multiple_sensor_data = np.reshape(multiple_sensor_readings, (sample_y, sample_x, len(sensor_ids)))
    
    coordinate_axes = (w+initial_position_mm[0], h+initial_position_mm[1])
    max_position_list_mm = []
    for i in range(len(sensor_ids)):
        max_position_mm = (x_t[max_indices[i]], y_t[max_indices[i]], z_t)
        max_position_list_mm.append(max_position_mm)

    target_coordinates = (x_t, y_t, z_t) # z_t is not ndarray

    return multiple_sensor_data, coordinate_axes, max_position_list_mm, target_coordinates


def get_coarse_laser_positions(initial_position_mm, width_mm, height_mm, delta_mm, sensor_ids):
    
    multiple_sensor_data, coordinate_axes, max_position_list_mm, target_coordinates = search_for_multiple_laser_position(initial_position_mm, width_mm, height_mm, delta_mm, sensor_ids)
    

    return max_position_list_mm


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
    

class Multiple_Circle_Detector:
    def __init__(self, max_detection_count=3):
        self.max_detection_count = max_detection_count 
        self.previous_circles = []

    def get_surrounding_ellipse_points(self, detected_circle):
        points = []
        
        x = detected_circle.x
        y = detected_circle.y
        v0 = detected_circle.v0
        m0 = detected_circle.m0
        v1 = detected_circle.v1
        m1 = detected_circle.m1

        e = 0
        while e <= 2 * np.pi:
            fx = x + np.cos(e) * v0 * m0 * 2 + v1 * m1 * 2 * np.sin(e)
            fy = y + np.cos(e) * v1 * m0 * 2 - v0 * m1 * 2 * np.sin(e)
            fxi = (int)(fx + 0.5)
            fyi = (int)(fy + 0.5)
            if fxi >= 0 and fxi < self.img_width and fyi >= 0 and fyi < self.img_height:
                points.append([fxi, fyi])

            e += 0.05

        points = np.array(points, np.int32)
        return points

    def get_circle_coordinates(self, circles):
        """
            circles: list of detected circles
        """
        circle_coordinates = []
        for circle in circles:
            circle_coordinates.append([circle.x, circle.y])

        return circle_coordinates

    def detect_multiple_circles(self, image, ):
        """
            returns detected circle coordinates as a list
        """
        img = image.copy()
        self.img_height = img.shape[0]
        self.img_width = img.shape[1]

        print(img.shape)
        detected_circles = []
        i = 0

        detected_circle_objects = []
        while i < self.max_detection_count:
            if len(self.previous_circles) > 0:
                prevCircle = self.previous_circles.pop()
            else:
                prevCircle = CircleClass()

            circle_detector = CircleDetectorClass(self.img_width, self.img_height)
            detected_circle = circle_detector.detect_np(img, prevCircle)
            


            if detected_circle.x == 0 and detected_circle.y == 0:
                break

            detected_circle_objects.append(detected_circle)
            
                
            detected_circles.append([detected_circle.x, detected_circle.y])

            img = cv2.circle(
                img,
                (int(detected_circle.x), int(detected_circle.y)),
                radius=10,
                color=(0, 255, 0),
                thickness=2,
            )

            points = self.get_surrounding_ellipse_points(detected_circle=detected_circle)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(img, [points], (0,0,255))
            # plt.imshow(img)
            # plt.show()
            i += 1
        self.previous_circles = detected_circle_objects
        return detected_circle_objects, img


def find_distances_from_mirror_center(point_1_mm, point_2_mm, point_3_mm, distances_mm, eps_mm=0.01, is_colinear=True):
    """
    ------------------------------------------------------------------------------------------------------------------------------------------
    if is_colinear=True
        point 1 point 2 and point 3 must have same x distance on the target plate, target plate must be perpendicular to ground, laser must be parallel to ground
        Calibration plate point order should be:

        ---------------------
        |    p1             |
        |    p2             |
        |    p3             |
        ---------------------

    else: 
        point 1 and point 3 must have same x distance on the target plate, target plate must be perpendicular to ground, laser must be parallel to ground
        Calibration plate point order should be:

        ---------------------
        |    p1          p2 |
        |                   |
        |    p3             |
        ---------------------
    ---------------------------------------------------------------------------------------------------------------------------------------

    
    point_1_mm: list, 3d measured coordinate of laser
    point_2_mm: list, 3d measured coordinate of laser
    point_3_mm: list, 3d measured coordinate of laser
    distances_mm: real distances between points in calibration plate [ |p1-p2|, |p2-p3|, |p1-p3| ]
    eps_mm: distance(mm) between points to be counted as same not needed if is_colinear is True
    is_colinear: bool, choose between 2 modes 

    return: distances of each point in mm
    """
    p1 = np.array(point_1_mm)
    p2 = np.array(point_2_mm)
    p3 = np.array(point_3_mm)


    if is_colinear:
        z2_1 = distances_mm[0] * point_1_mm[2] / (np.linalg.norm(p1-p2))

        z2_2 = distances_mm[1] * point_2_mm[2] / (np.linalg.norm(p2-p3))

        z2_3 = distances_mm[2] * point_3_mm[2] / (np.linalg.norm(p1-p3))

        avg_z = (z2_1 + z2_2 + z2_3) / 3
        return avg_z, avg_z, avg_z
    
    else:

        z2 = distances_mm[2] * point_1_mm[2] / (np.linalg.norm(p1-p3))
        z0 = point_1_mm[2]

        g = p1 * z2 / z0

        # for points p1, p2
        a = (p2[0] / z0)**2 + (p2[1] / z0)**2 + (p2[2] / z0)**2
        b = -2*g[0]*p2[0]/z0 -2*g[1]*p2[1]/z0 -2*g[2]*p2[2]/z0
        c = g[0]**2 + g[1]**2 + g[2]**2 - distances_mm[0]**2


        x1_f, x2_f = quadratic_solver(a, b, c)

        print("p1-p2")
        print(x1_f)
        print(x2_f)

        g = p3 * z2 / z0
        # for points p2, p3
        a = (p2[0] / z0)**2 + (p2[1] / z0)**2 + (p2[2] / z0)**2
        b = -2*g[0]*p2[0]/z0 -2*g[1]*p2[1]/z0 -2*g[2]*p2[2]/z0
        c = g[0]**2 + g[1]**2 + g[2]**2 - distances_mm[1]**2
        x1_s, x2_s = quadratic_solver(a, b, c)

        print("p2-p3")
        print(x1_s)
        print(x2_s)
        
        if abs(x1_f - x1_s) <= eps_mm:
            z3 = (x1_f + x1_s)/2
        elif abs(x1_f - x2_s) <= eps_mm:
            z3 = (x1_f + x2_s)/2
        elif abs(x2_f - x1_s) <= eps_mm:
            z3 = (x2_f + x1_s) /2
        elif abs(x2_f - x2_s) <= eps_mm:
            z3 = (x2_f + x2_s) / 2
        else:
            print(f"Obtained distances are not within {eps_mm} mm.")

        return (z2, z3, z2)

def quadratic_solver(a, b, c):
    x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2*a)
    x2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2*a)

    return x1, x2

def identify_points(point1, point2, point3, is_colinear=True):
    """
            ------------------------------------------------------------------------------------------------------------------------------------------
    if is_colinear=True
        point 1 point 2 and point 3 must have same x distance on the target plate, target plate must be perpendicular to ground, laser must be parallel to ground
        Calibration plate point order should be:

        ---------------------
        |    p1             |
        |    p2             |
        |    p3             |
        ---------------------

    else: 
        point 1 and point 3 must have same x distance on the target plate, target plate must be perpendicular to ground, laser must be parallel to ground
        Calibration plate point order should be:

        ---------------------
        |    p1          p2 |
        |                   |
        |    p3             |
        ---------------------
    ---------------------------------------------------------------------------------------------------------------------------------------

        Identify points and order them according to their position on calibration plate
       
        point1: list [x, y, z]
        point2: list [x, y, z]
        point3: list [x, y, z]
    """

    if is_colinear:

        points = [point1, point2, point3]
        point_y = []
        point_y.append(point1[1])
        point_y.append(point2[1])
        point_y.append(point3[1])

        sorted_args = argsort(point_y)

        return points[sorted_args[0]], points[sorted_args[1]], points[sorted_args[2]] # p1, p2, p3


        

    else:
        points = [point1, point2, point3]

        max_x = -100000000000000000 # arbitrary small number
        max_x_idx = None
        for i, p in enumerate(points):
            if p[0] >= max_x:
                max_x = p[0]
                max_x_idx = i

        idx_set = set([0, 1, 2])
        idx_set.remove(max_x_idx)
        idx_list = list(idx_set)
        idx_1 = idx_list[0]
        idx_2 = idx_list[1]

        if points[idx_1][1] >= points[idx_2][1]:
            return points[idx_2], points[max_x_idx], points[idx_1]  # p1, p2, p3
        else:
            return points[idx_1], points[max_x_idx], points[idx_2]  # p1, p2, p3


def get3d_coords_from_pixel_coords(pix_coords, transformed_depth_img):
    pix_x, pix_y = pix_coords
    pix_x = int(pix_x)
    pix_y = int(pix_y)
    
    rgb_depth = transformed_depth_img[pix_y, pix_x]
    pixels = k4a_float2_t((pix_x, pix_y))
    pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    coordinates_3d = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))
    return coordinates_3d


def get_sensor_pos_from_marker_pos(marker_positions, distance_of_sensor_from_marker_mm, distance_of_second_sensor_from_first_sensor_mm):
    """
        transformed_depth_img: depth image transformed into color image coordinate system
        marker_positions: 3d camera positions of markers (3D)
        distance_of_sensor_from_marker_mm: float, position of first sensor relative to first marker (- if sensor is on the left, + if sensor is on the right) 
        distance_of_second_sensor_from_first_sensor_mm: float 


        Sensor placement:
         ---------------------
        |    p1             |
        |    p2             |
        |    p3             |
        ---------------------
        Marker placement:


    """
    center_pos_3d = np.mean(marker_positions, axis=1)

    r_vec, d_vec = extract_unit_vectors(marker_positions)
    d_vec = np.array([0, 1, 0]) # due to geometry of calibration plate

    sensor_pos_2 = center_pos_3d + r_vec * distance_of_sensor_from_marker_mm
    sensor_pos_1 = sensor_pos_2 - d_vec * distance_of_second_sensor_from_first_sensor_mm
    sensor_pos_3 = sensor_pos_2 + d_vec * distance_of_second_sensor_from_first_sensor_mm

    return sensor_pos_1, sensor_pos_2, sensor_pos_3, r_vec, d_vec


def record_color_and_depth_image():
    while True:
        capture = device.update()
        ret_color, color_image = capture.get_color_image() 

        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
        if not ret_color or not ret_depth:
            continue  

        if cv2.waitKey(1) == ord('s'):
            cv2.imwrite("color_im.png", color_image)
            cv2.imwrite("transformed_depth_img.png", transformed_depth_image)
        
        cv2.imshow("image", color_image)
        #cv2.imshow("image", transformed_depth_image)
        if cv2.waitKey(1) == ord('q'):
            break


def extract_unit_vectors(marker_positions, marker_size=(9, 6)):
    down_vec = 0
    for i in range(marker_size[1]):        
        for j in range(marker_size[0]-1):
            idx_up = j + i * marker_size[0]
            idx_down = 1+j + i * marker_size[0]
            down_vec += (marker_positions[:, idx_down] - marker_positions[:, idx_up])

    down_vec /= ( (marker_size[0]-1) * marker_size[1] )



    right_vec = 0
    for i in range(marker_size[1]-1):        
        for j in range(marker_size[0]):
            idx_left = j + (i+1) * marker_size[0]
            idx_right = j + i * marker_size[0]
            right_vec += (marker_positions[:, idx_right] - marker_positions[:, idx_left])

    right_vec /= ( marker_size[0] * (marker_size[1]-1) )

    right_vec /= np.linalg.norm(right_vec)
    down_vec /= np.linalg.norm(down_vec)


    return right_vec, down_vec


def set_initial_laser_pos(initial_z=540):


    def update_mirror(event):
        global mirror_x, mirror_y, mirror_z
        y_t = np.array([w2.get()])
        x_t = np.array([w1.get()])
        coordinate_transform = CoordinateTransform(d=0, D=w3.get(), rotation_degree=45)
        y_m, x_m = coordinate_transform.target_to_mirror(y_t, x_t) # order is changed in order to change x and y axis
        si_0.SetXY(y_m[0])        
        si_1.SetXY(x_m[0]) 
        mirror_x, mirror_y, mirror_z = w1.get(), w2.get(), w3.get()



    def increaseX():    
        w1.set(w1.get()+1)
        

    def decreaseX():    
        w1.set(w1.get()-1)
        

    def increaseY():    
        w2.set(w2.get()+1)
        

    def decreaseY():   
        w2.set(w2.get()-1)
        

    def increaseZ():
        w3.set(w3.get()+1)

    def decreaseZ():
        w3.set(w3.get()-1)

    master = tk.Tk()
    w1 = tk.Scale(master, from_=-500, to=500, tickinterval=1, command=update_mirror)
    w1.set(0)
    w1.pack()
    tk.Button(master, text='Increase X', command=increaseX).pack()
    tk.Button(master, text='Decrease X', command=decreaseX).pack()



    w2 = tk.Scale(master, from_=-500, to=500,tickinterval=1, command=update_mirror)
    w2.set(0)
    w2.pack()
    tk.Button(master, text='Increase Y', command=increaseY).pack()
    tk.Button(master, text='Decrease Y', command=decreaseY).pack()


    w3 = tk.Scale(master, from_=20, to=1000,tickinterval=1,  command=update_mirror)
    w3.set(initial_z)
    w3.pack()
    tk.Button(master, text='Increase Z', command=increaseZ).pack()
    tk.Button(master, text='Decrease Z', command=decreaseZ).pack()

    

    master.mainloop()

    return mirror_x, mirror_y, mirror_z  # x, y, z position of laser

def update_laser_position(old_point, z_new):
    """
        old_point: list length 3 
        z_new: float

        updates measured laser position by using calculated distance from mirror center (z_new) and initial distance assumption (z_old)
    """
    scale = z_new / old_point[2]

    new_point = scale * np.array(old_point)

    return new_point


def calibrate(width_mm, height_mm, delta_mm, sensor_ids):
    laser_points = []
    camera_points = []
    num_iter=0
    previous_p2_pos = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while num_iter < ITER_COUNT:
        #################################### find location of laser detectors using depth camera ##################################################################
        multiple_circle_detector = Multiple_Circle_Detector()

        num_color_img = 0
        avg_points_cam_3d = np.zeros((3, 9*6))
        

        while num_color_img < CAPTURE_COUNT:

            capture = device.update()
            ret_color, color_image = capture.get_color_image() 

            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
            if not ret_color or not ret_depth:
                continue  

            color_image_orig = color_image.copy()
            # color_image = cv2.imread("circle-medium.png")
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            # If found, add object points, image points (after refining them)
            if not ret:
                print("Chessboard not detected. Move the board and press enter.")
                input()
                continue  
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            # Draw and display the corners
            cv2.drawChessboardCorners(color_image, (9,6), corners2, ret)
            
            cv2.putText(color_image, f"Current Iteration: {num_iter+1}/{ITER_COUNT}", (10, 40), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
            

            
            print("\n\nPress (y) to include image. Press (n) to discard image.")
            cv2.putText(color_image, "Press (y) to include image. Press (n) to discard image.", (10, 80), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("image",  cv2.resize(color_image, (1280, 720)))
            if cv2.waitKey(0) == ord('y'):
                num_color_img += 1

                
                for i, point in enumerate(corners2):          
                    point_cam_3d = get3d_coords_from_pixel_coords((point[0, 0], point[0, 1]), transformed_depth_img=transformed_depth_image).reshape((-1))
                    avg_points_cam_3d[:, i] += point_cam_3d     

                
      
            

            if cv2.waitKey(1) == ord('q'):
                sys.exit()
        

        avg_points_cam_3d /= CAPTURE_COUNT               


        sensor_pos_cam_1, sensor_pos_cam_2, sensor_pos_cam_3, r_vec, d_vec = get_sensor_pos_from_marker_pos(avg_points_cam_3d, distance_of_sensor_from_marker_mm=SENSOR_POS_WRT_MARKER, distance_of_second_sensor_from_first_sensor_mm=SENSOR_DISTANCE)

        
        # deduce new position by using marker pattern if it is possible
        if num_iter > 1:
            

            current_camera_points_np = avg_points_cam_3d

            # diff = current_camera_points_np - previous_camera_points_np
            # average_diff = np.mean(diff, axis=1)

            # initial_search_point = previous_p2_pos.reshape((3, 1)) + average_diff.reshape((3, 1))


            R, t = optimal_rotation_and_translation(previous_camera_points_np, current_camera_points_np)

            
            previous_p2_pos_in_camera_coords = R_temp.T @ (previous_p2_pos.reshape((3, 1)) - t_temp)
            initial_search_point_in_camera_coords = R @ previous_p2_pos_in_camera_coords + t
            initial_search_point = R_temp @ initial_search_point_in_camera_coords + t_temp

            x_init, y_init, z_init = initial_search_point[0, 0], initial_search_point[1, 0], initial_search_point[2, 0]

            p2_sensor_estimate = np.array([x_init, y_init, z_init])

            p1_sensor_estimate = p2_sensor_estimate - d_vec.reshape((-1)) * SENSOR_DISTANCE
            p3_sensor_estimate = p2_sensor_estimate + d_vec.reshape((-1)) * SENSOR_DISTANCE         

            coarse_laser_pos = [p1_sensor_estimate, p2_sensor_estimate, p3_sensor_estimate] 
           

        else:
            x_init, y_init, z_init = set_initial_laser_pos()

            coarse_laser_pos = get_coarse_laser_positions([x_init, y_init, z_init], width_mm, height_mm, delta_mm, sensor_ids)

        #################################### find location of infrared detectors using laser #######################################################
        
        #x_init, y_init, z_init = set_initial_laser_pos()

        

        coarse_laser_pos = get_fine_laser_positions(coarse_laser_pos, search_length_mm=30, delta_mm=2)
        fine_laser_coords = get_fine_laser_positions(coarse_laser_pos, search_length_mm=10, delta_mm=0.5)

        p1, p2, p3 = identify_points(fine_laser_coords[0], fine_laser_coords[1], fine_laser_coords[2])

        # adjust distances according to calibration plat geometry (assumed 100 mm spacing)
        p1_z, p2_z, p3_z = find_distances_from_mirror_center(point_1_mm=p1, point_2_mm=p2, point_3_mm=p3, distances_mm=[75, 75, 150]) # all of  the zs are same

        p1_updated = update_laser_position(old_point=p1, z_new=p1_z)
        p2_updated = update_laser_position(old_point=p2, z_new=p2_z)
        p3_updated = update_laser_position(old_point=p3, z_new=p3_z)

        updated_points = [p1_updated, p2_updated, p3_updated]

        real_3d_coords = []
        real_3d_coords.append(p1_updated.reshape((-1)))
        real_3d_coords.append(p2_updated.reshape((-1)))
        real_3d_coords.append(p3_updated.reshape((-1)))



        # laser detector positions in laser mirror coordinate system
        # real_3d_coords = []

        # for i in range(len(updated_points)):
        #     sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=[updated_points[i][0], updated_points[i][1], updated_points[i][2]], width_mm=50, height_mm=50, delta_mm=2, sensor_id=i+1)
        #     sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=max_pos, width_mm=10, height_mm=10, delta_mm=0.2, sensor_id=i+1)

        #     real_3d_coords.append(max_pos)

        print(f"Press (s) to save measurements. Press another character to discard measurements in current iteration." )
        cv2.putText(color_image_orig, f"Current Iteration: {num_iter+1}/{ITER_COUNT}", (10, 40), font, 2, (0, 255, 0),3, cv2.LINE_AA)
        cv2.putText(color_image_orig, "Press (s) to save measurements, another character to discard measurements in current iteration.", (10, 80), font, 2, (0, 255, 0), 3, cv2.LINE_AA)
        
        
        cv2.imshow("image", cv2.resize(color_image_orig, (1280, 720)))
        key = cv2.waitKey(0)
        if  key== ord('s'):
            camera_points.append(sensor_pos_cam_1.reshape((-1)))
            camera_points.append(sensor_pos_cam_2.reshape((-1)))
            camera_points.append(sensor_pos_cam_3.reshape((-1)))
            laser_points.extend(real_3d_coords)     
            previous_p2_pos = p2_updated   
            previous_camera_points_np = avg_points_cam_3d

            laser_points_temp_np = np.array(laser_points).T
            camera_points_temp_np = np.array(camera_points).T

            R_temp, t_temp = optimal_rotation_and_translation(camera_points_temp_np, laser_points_temp_np)            

            num_iter+=1
        
        



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

        with open('{}/parameters.pkl'.format(CALIBRATION_SAVE_PATH, ITER_COUNT), 'wb') as f:
            pickle.dump(calibration_dict, f)


    # print(f"Point1 Laser: {real_3d_coords[0]}")
    # print(f"Point2 Laser: {real_3d_coords[1]}")
    # print(f"Point3 Laser: {real_3d_coords[2]}")


    





################################################### TESTS ############################################

def test_identify_points():
    print(identify_points(point1=[5, 10, 20], point2=[-2, 1, 6], point3=[50, 10, 30]))


def test_search_laser_positon():

    sensor_id = 2
    sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=[0, 0, 400], width_mm=50, height_mm=100, delta_mm=2, sensor_id=sensor_id)


    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()



    sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=max_pos, width_mm=10, height_mm=10, delta_mm=0.2, sensor_id=sensor_id)
    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()

def test_detect_multiple_circles():
    ret_color = False
    multiple_circle_detector = Multiple_Circle_Detector(max_detection_count=3)
    while True:
        capture = device.update()
        ret_color, color_image = capture.get_color_image() 
        if not ret_color:
            continue
        # color_image = cv2.imread("circle-medium.png")
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, axis=2)
        gray = np.concatenate([gray, gray, gray], axis=2)

        detected_circles, marked_image = multiple_circle_detector.detect_multiple_circles(gray)   
        circle_coordinates = multiple_circle_detector.get_circle_coordinates(detected_circles)
        circle_coordinates = np.array(circle_coordinates)

        for circle in circle_coordinates:
            cv2.circle(gray, center=(int(circle[0]), int(circle[1])), radius=10, color=(0, 255, 0), thickness=2)
        cv2.imshow("image", gray)
        if cv2.waitKey(1) == ord('q'):
            break


def test_get_coarse_laser_positions():
    print("Coarse laser positions:")
    print(get_coarse_laser_positions(initial_position_mm=[0, 0, 50], width_mm=500, height_mm=500, delta_mm=2, sensor_ids=[1,2,3]))


def test_search_for_multiple_laser_position():
    multiple_sensor_data, coordinate_axes, max_position_list_mm, target_coordinates = search_for_multiple_laser_position(initial_position_mm=[-40, 60, 500], width_mm=210, height_mm=180, delta_mm=3, sensor_ids=[1, 2, 3])

    print(max_position_list_mm)   


    for i in range(3):
        plt.figure()
        plt.imshow(multiple_sensor_data[:,:,i])
        
    plt.show()

def test_sensor_reading():
    while True:
        print("Sensor 1: ", get_sensor_reading(1))
        time.sleep(0.05)
        print("Sensor 2: ", get_sensor_reading(2))
        time.sleep(0.05)
        print("Sensor 3: ", get_sensor_reading(3))
        time.sleep(1)


if __name__ == "__main__":
    calibrate(width_mm=60, height_mm=240, delta_mm=3, sensor_ids=[1, 2, 3])


