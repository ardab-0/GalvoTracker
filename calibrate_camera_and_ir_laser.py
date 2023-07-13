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
    s.flush()
    s.write("pd\n".encode())
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

        time.sleep(0.005)
        sensor_readings.append(get_sensor_reading(0))

    sensor_readings = np.array(sensor_readings)
    max_idx = np.argmax(sensor_readings)

    return np.reshape(sensor_readings, (sample_y, sample_x)), (w+initial_position_mm[0], h+initial_position_mm[1]), (x_t[max_idx], y_t[max_idx], z_t)

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




def test_search_laser_positon():
    sensor_data, (width_range, height_range), max_pos = search_for_laser_position(initial_position_mm=[0, -20, 500], width_mm=50, height_mm=50, delta_mm=2)


    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()



    sensor_data, (width_range, height_range), max_pos = search_for_laser_position(initial_position_mm=max_pos, width_mm=10, height_mm=10, delta_mm=0.2)
    print(max_pos)

    plt.imshow(sensor_data, extent=[width_range[0], width_range[-1], height_range[-1], height_range[0]])
    plt.show()




test_detect_multiple_circles()