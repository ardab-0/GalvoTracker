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
from utils import optimal_rotation_and_translation
import pickle
import sys
import matplotlib.pyplot as plt
from image_processing.local_maxima_finding import find_local_maxima

# Constants
d = 0
mirror_rotation_deg = 45
num_iterations = 50
save_path = "calibration_parameters"


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
s = serial.Serial(port="COM6", parity=serial.PARITY_EVEN, stopbits=serial.STOPBITS_ONE, timeout=1)



def get_sensor_reading(sensor_id): 
    """
    sensor_id: int
    """   
    s.flush()
    s.write(f"pd_{sensor_id}\n".encode())
    mes = s.read_until()
    return float(mes.decode().strip("\r\n"))


def search_for_laser_position(initial_position_mm, width_mm, height_mm, delta_mm):
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

        time.sleep(0.002)
        sensor_readings.append(get_sensor_reading(0))

    sensor_readings = np.array(sensor_readings)
    max_idx = np.argmax(sensor_readings)

    sensor_data = np.reshape(sensor_readings, (sample_y, sample_x))
    coordinate_axes = (w+initial_position_mm[0], h+initial_position_mm[1])
    max_position_mm = (x_t[max_idx], y_t[max_idx], z_t)
    target_coordinates = (x_t, y_t, z_t) # z_t is not ndarray


    return sensor_data, coordinate_axes, max_position_mm, target_coordinates


def get_coarse_laser_positions(initial_position_mm, width_mm, height_mm, delta_mm):
    """
    
    """
    def convert_coord_to_idx(coord, col):
        return coord[1] * col + coord[0] 


    sensor_data, (width_range, height_range), max_pos, target_coordinates = search_for_laser_position(initial_position_mm, width_mm, height_mm, delta_mm)
    x_t, y_t, z_t = target_coordinates


    # normalize sensor_data
    sensor_data = sensor_data / np.max(sensor_data)

    image_coords = find_local_maxima(sensor_data, number_of_maxima=3)
    coords_3d = []

    for im_coord in image_coords:
        idx = convert_coord_to_idx( im_coord, col=len(sensor_data[0]) )
        x = x_t[idx]
        y = y_t[idx]
        z = z_t
        coords_3d.append([x, y, z]) 

    return coords_3d


def get_fine_laser_positions(rough_laser_coords):
    """
        rough_laser_coords: list
    """

    fine_coords_3d = []
    for coord in rough_laser_coords: 
        sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=coord, width_mm=10, height_mm=10, delta_mm=0.2)
        fine_coords_3d.append(max_pos)

    # plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    # plt.show()

    return fine_coords_3d
    

class Multiple_Circle_Detector:
    def __init__(self, max_detection_count=5):
        self.max_detection_count = max_detection_count 
        self.previous_circles = []

    def detect_multiple_circles(self, image, ):
        """
            returns detected circle coordinates as a list
        """
        img = image.copy()
        img_height = img.shape[0]
        img_width = img.shape[1]

        print(img.shape)
        detected_circles = []
        i = 0

        detected_circle_objects = []
        while i < self.max_detection_count:
            if len(self.previous_circles) > 0:
                prevCircle = self.previous_circles.pop()
            else:
                prevCircle = CircleClass()

            circle_detector = CircleDetectorClass(img_width, img_height)
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
                if fxi >= 0 and fxi < img_width and fyi >= 0 and fyi < img_height:
                    points.append([fxi, fyi])

                e += 0.05

            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(img, [points], (0,0,255))
            # plt.imshow(img)
            # plt.show()
            i += 1
        self.previous_circles = detected_circle_objects
        return detected_circles, img



# not working
# def find_distances(points_1_mm, points_2_mm, distances_mm):

#     A = np.zeros((len(points_1_mm), 3))
#     y = np.array(distances_mm)
#     y = np.square(y)
#     y = y.reshape((-1, 1))

#     for i in range(len(points_1_mm)):
#         a = (points_1_mm[i][0] / points_1_mm[i][2])**2 + (points_1_mm[i][1] / points_1_mm[i][2])**2 + (points_1_mm[i][2] / points_1_mm[i][2])**2

#         b = -(points_1_mm[i][0] * points_2_mm[i][0]) / points_1_mm[i][2]**2 -(points_1_mm[i][1] * points_2_mm[i][1]) / points_1_mm[i][2]**2 -(points_1_mm[i][2] * points_2_mm[i][2]) / points_1_mm[i][2]**2

#         c = (points_2_mm[i][0] / points_1_mm[i][2])**2 + (points_2_mm[i][1] / points_1_mm[i][2])**2 + (points_2_mm[i][2] / points_1_mm[i][2])**2


#         A[i, 0] = a
#         A[i, 1] = b
#         A[i, 2] = c


#     x = np.linalg.pinv(A) @ b

#     z1 = np.sqrt(x[0])
#     z2 = np.sqrt(x[2])

#     return z1, z2


def find_distances(point_1_mm, point_2_mm, point_3_mm, distances_mm, eps=0.01 ):
    p1 = np.array(point_1_mm)
    p2 = np.array(point_2_mm)
    p3 = np.array(point_3_mm)

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
    
    if abs(x1_f - x1_s) <= eps:
        z3 = x1_f
    elif abs(x1_f - x2_s) <= eps:
        z3 = x1_f
    elif abs(x2_f - x1_s) <= eps:
        z3 = x2_f
    elif abs(x2_f - x2_s) <= eps:
        z3 = x2_f
    else:
        print(f"Obtained distances are not within {eps}")

    return z2, z3

def quadratic_solver(a, b, c):
    x1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2*a)
    x2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2*a)

    return x1, x2



################################################### TESTS ############################################


def test_search_laser_positon():
    sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=[0, -20, 500], width_mm=50, height_mm=50, delta_mm=2)


    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()



    sensor_data, (width_range, height_range), max_pos, _ = search_for_laser_position(initial_position_mm=max_pos, width_mm=10, height_mm=10, delta_mm=0.2)
    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()

def test_detect_multiple_circles():
    ret_color = False
    multiple_circle_detector = Multiple_Circle_Detector()
    while True:
        capture = device.update()
        ret_color, color_image = capture.get_color_image() 
        if not ret_color:
            continue
        # color_image = cv2.imread("circle-medium.png")
        circle_coordinates, marked_image = multiple_circle_detector.detect_multiple_circles(color_image)   
        print(circle_coordinates)
        cv2.imshow("image", marked_image)
        if cv2.waitKey(1) == ord('q'):
            break


def test_get_coarse_laser_positions():
    print("Coarse laser positions:")
    print(get_coarse_laser_positions(initial_position_mm=[0, 0, 50], width_mm=500, height_mm=500, delta_mm=2))



print(find_distances(point_1_mm=[20, 20, 10], point_2_mm=[30, 30, 10], point_3_mm=[20, 30, 10], distances_mm=[170.88, 135.65, 60]))

# test_detect_multiple_circles()