import cv2
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
from image_processing.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import pickle
import time
from pykinect_azure.k4a.transformation import Transformation



# Constants 
d = 0
mirror_rotation_deg = 45
save_path = "ir_calibration_parameters"
# target coordinate offset (mm)


with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]

# initial mouse position
mouse_x = 500
mouse_y = 300

def onMousemove(event, x, y, flags, param):
	pass
	
mul = 1
def updatePos(dt):
    global mouse_x, mouse_y, mul
    
    mouse_x = mouse_x + dt*400*mul
    if mouse_x > 900:
         mul = -1
    elif mouse_x < 200:
         mul = 1
     #mouse_y = mouse_y + dt*5

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

    transformation = Transformation(device.get_calibration(device_config.depth_mode, device_config.color_resolution))


    cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Laser Detector', onMousemove)


    next_t = 0.085
    prev_point = 0
    prev_v = 0
    start = time.time()

    while True:
        
        # Get capture
        capture = device.update()

        # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()

        # Get the colored depth
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()
        

        if not ret_color or not ret_depth:
            continue  

        
        pix_x = int(mouse_x)
        pix_y = int(mouse_y)
        color_image = cv2.circle(color_image, (pix_x, pix_y), radius=10, color=(0, 255, 0), thickness=2)

        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))



        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
        # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))

    
        # Test point cloud 
        # point_cloud_image = transformation.depth_image_to_point_cloud(transformed_depth_image, K4A_CALIBRATION_TYPE_COLOR)
        

        
        # rotate and translate

        camera_coordinates_in_laser_coordinates =  R @ camera_coordinates + t
        now = time.time()
        dt = now - start
        
        # update mouse position
        updatePos(dt)

        start = now
        v = (camera_coordinates_in_laser_coordinates - prev_point) / dt
        v_avg = 0.5*v + 0.5*prev_v
        prev_v = v_avg
        prev_point = camera_coordinates_in_laser_coordinates

        prediction_coor = camera_coordinates_in_laser_coordinates + v_avg*next_t

        # camera_coordinates_in_laser_coordinates = np.floor(camera_coordinates_in_laser_coordinates)

        # print("camera_coordinates", camera_coordinates)

        # print("camera_coordinates_in_laser_coordinates", camera_coordinates_in_laser_coordinates)


        coordinate_transform = CoordinateTransform(d=d, D=prediction_coor[2], rotation_degree=mirror_rotation_deg)



        y_m, x_m = coordinate_transform.target_to_mirror(prediction_coor[1], prediction_coor[0]) # order is changed in order to change x and y axis

        
        
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0]) 


        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(color_image, f"fps: {1 / (time.time() - start)}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # cv2.putText(color_image, f"Target Coordinates w.r.t. mirror center:", (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"X: {camera_coordinates_in_laser_coordinates[0]}", (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"Y: {camera_coordinates_in_laser_coordinates[1]}", (10, 80), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"Z: {camera_coordinates_in_laser_coordinates[2]}", (10, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        
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