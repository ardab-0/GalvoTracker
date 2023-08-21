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
save_path = "ir_calibration_parameters_test"
# target coordinate offset (mm)
lower_red = np.array([140,   10, 240]) 
upper_red = np.array([180, 130, 256])
CHESSBOARD_WIDTH = 9
CHESSBOARD_LENGTH = 6


with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    R = loaded_dict["R"]
    t = loaded_dict["t"]

# initial mouse position
mouse_x = 0
mouse_y = 0

def onMousemove(event, x, y, flags, param):
	global mouse_x, mouse_y
	if event == cv2.EVENT_MOUSEMOVE:
		mouse_x = x
		mouse_y = y

def get3d_coords_from_pixel_coords(pix_coords, transformed_depth_img, device):
    pix_x, pix_y = pix_coords
    pix_x = int(pix_x)
    pix_y = int(pix_y)
    
    rgb_depth = transformed_depth_img[pix_y, pix_x]
    pixels = k4a_float2_t((pix_x, pix_y))
    pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
    coordinates_3d = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))
    return coordinates_3d

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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Laser Detector', onMousemove)

    ret = False

    while not ret:             
        capture = device.update()        
        ret_color, color_image = capture.get_color_image()       
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image() 
        if not ret_color or not ret_depth:
            continue 
        color_image_display = color_image.copy()
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_WIDTH, CHESSBOARD_LENGTH), None)
    
    corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    #cv2.drawChessboardCorners(color_image_display, (9,6), corners2, ret)

    centers = []
    
    for j in range(CHESSBOARD_LENGTH - 1):
        for i in range(CHESSBOARD_WIDTH - 1):
            tl = i + j * CHESSBOARD_WIDTH
            tr = tl + 1
            bl = tl + CHESSBOARD_WIDTH
            br = bl + 1 
            center = corners[tl] + corners[tr] + corners[bl] + corners[br]
            center /= 4

            centers.append(center)

    
    

    for i, point in enumerate(corners2):          
        point_cam_3d = get3d_coords_from_pixel_coords((point[0, 0], point[0, 1]), transformed_depth_img=transformed_depth_image, device=device).reshape((-1))       
        # rotate and translate
        camera_coordinates_in_laser_coordinates =  R @ point_cam_3d.reshape((3, 1)) + t        
        coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)
        y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1], camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis
                
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0])    

        ret_color = False
        while not ret_color:             
            capture = device.update()        
            ret_color, color_image = capture.get_color_image()     

        color_image_display = color_image.copy()    
        circle = detect_circle_position(color_image, lower_range=lower_red, upper_range=upper_red)

        while circle is  None:
            capture = device.update()        
            ret_color, color_image = capture.get_color_image() 
            if not ret_color:
                    continue
            color_image_display = color_image.copy()

            circle = detect_circle_position(color_image, lower_range=lower_red, upper_range=upper_red)



        circle = np.round(circle).astype("int")
        cv2.circle(color_image_display, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)

        pix_x = circle[0]
        pix_y = circle[1]
        rgb_depth = transformed_depth_image[pix_y, pix_x]

        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)       
        laser_in_camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z])

        rmse = np.sqrt(np.mean(np.square(point_cam_3d.reshape((-1)) - laser_in_camera_coordinates)))          
    
        print("RMSE: ", rmse)  
        cv2.imshow('Laser Detector',color_image_display)
        cv2.waitKey(0)



    mre2.disconnect()
    print("done")



if __name__ == "__main__":
     main()