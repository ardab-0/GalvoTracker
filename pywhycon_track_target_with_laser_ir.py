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

import pykinect_azure as pykinect
import cv2
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure import Image
from pykinect_azure.k4a import _k4a
import time
from circle_detector_library.circle_detector_module import *

# Constants 

d = 0
mirror_rotation_deg = 45
save_path = "calibration_parameters"
DOWNSAMPLE = 4


with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]


class BufferImage(Image):
    def __init__(self, image_handle=None):
        super().__init__(image_handle=image_handle)

    @staticmethod
    def create_from_buffer(image_format, width_pixels, height_pixels, stride_bytes, buffer, buffer_size, buffer_release_cb, buffer_release_cb_context):
        handle = _k4a.k4a_image_t()
        #_k4a.k4a_image_create_from_buffer(image_format=image_format, width=width_pixels, height=height_pixels, stride=stride_bytes, buffer=buffer,
        #                                  buffer_size=buffer_size, buffer_release_cb=buffer_release_cb, buffer_release_cb_context=buffer_release_cb_context, image_handle=handle)
        _k4a.VERIFY(_k4a.k4a_image_create_from_buffer(image_format=image_format, width=width_pixels, height=height_pixels, stride=stride_bytes, buffer=buffer, buffer_size=buffer_size, buffer_release_cb=buffer_release_cb, buffer_release_cb_context=buffer_release_cb_context, image_handle=handle),"Create image failed!")

        return Image(handle)



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
    device_config.synchronized_images_only = False

    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    transformation = Transformation(device.get_calibration(
        device_config.depth_mode, device_config.color_resolution))

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.namedWindow('IR Laser Tracker', cv2.WINDOW_NORMAL)

    # gives undefined warning but works (pybind11 c++ module) change import *
    prevCircle = CircleClass()
    circle_detector = CircleDetectorClass(int(1280/DOWNSAMPLE), int(720/DOWNSAMPLE)) # K4A_COLOR_RESOLUTION_720P

    while True:
        start = time.time()
        # Get capture
        capture = device.update()
        print("Time until capture (s): ", time.time() - start)

        # Get the color image from the capture
        # ret_color, color_image = capture.get_color_image()
        # print("Time until color image (s): ", time.time() - start)
        ir_image_obj = capture.get_ir_image_object()
        depth_img_obj = capture.get_depth_image_object()


        

        custom_ir_image = BufferImage.create_from_buffer(_k4a.K4A_IMAGE_FORMAT_CUSTOM16,
                                                        ir_image_obj.get_width_pixels(),
                                                        ir_image_obj.get_height_pixels(),
                                                        ir_image_obj.get_stride_bytes(),
                                                        ir_image_obj.get_buffer(),
                                                        ir_image_obj.get_height_pixels()*ir_image_obj.get_stride_bytes(),
                                                        None,
                                                        None)

        print("Time until custom ir image (s): ", time.time() - start)

        # modified by ardab to return transformed_depth_image 
        transformed_depth_image_obj, transformed_ir_image_obj = transformation.depth_image_to_color_camera_custom(depth_image=depth_img_obj, custom_image=custom_ir_image)

        print("Time until end of transform (s): ", time.time() - start)


        ret_transformed_ir, transformed_ir_image = transformed_ir_image_obj.to_numpy()
        ret_transformed_depth, transformed_depth_image = transformed_depth_image_obj.to_numpy()


        if not ret_transformed_ir or not ret_transformed_depth:
            continue
        
        transformed_ir_image = transformed_ir_image[::DOWNSAMPLE, ::DOWNSAMPLE]
        transformed_ir_image[transformed_ir_image>1000] = 1000
        transformed_ir_image = transformed_ir_image / 1000 * 255
        transformed_ir_image = transformed_ir_image.astype("uint8")
        transformed_ir_image = cv2.cvtColor(transformed_ir_image,cv2.COLOR_GRAY2RGB)
        print("Time until transformed ir image (s): ", time.time() - start)

        new_circle = circle_detector.detect_np(transformed_ir_image, prevCircle)    
        prevCircle = new_circle
        transformed_ir_image = cv2.circle(transformed_ir_image, (int(new_circle.x), int(new_circle.y)), radius=10, color=(0, 255, 0), thickness=2)


        pix_x = int(new_circle.x * DOWNSAMPLE)
        pix_y = int(new_circle.y * DOWNSAMPLE)
        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
        # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))

        # rotate and translate  
        camera_coordinates_in_laser_coordinates =  R @ camera_coordinates + t
    

        coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)
        y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1], camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis
        print("Time until completing calculations (s): ", time.time() - start)

        
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0])        


        end = time.time()
        print("elapsed time: ", (end - start))    
        cv2.putText(transformed_ir_image, f"fps: {1 / (end - start)}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(transformed_ir_image, f"Target Coordinates w.r.t. mirror center:", (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(transformed_ir_image, f"X: {camera_coordinates_in_laser_coordinates[0]}", (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(transformed_ir_image, f"Y: {camera_coordinates_in_laser_coordinates[1]}", (10, 80), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(transformed_ir_image, f"Z: {camera_coordinates_in_laser_coordinates[2]}", (10, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('IR Laser Tracker',transformed_ir_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break



    
if __name__ == "__main__":
    main()