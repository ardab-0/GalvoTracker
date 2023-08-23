import cv2
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t, k4a_float3_t
import numpy as np
from image_processing.circle_detector import detect_circle_position
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import pickle
import time
from pykinect_azure.k4a.transformation import Transformation
from utils import optimal_rotation_and_translation



# Constants 
d = 0
mirror_rotation_deg = 45
save_path = "ir_calibration_parameters_test"
num_iterations = 10
accuracy_path = "scanning_line/calibration_accuracy_423mm"
CALIBRATION_ITER = 8
sample_x = 10
sample_y = 10
lower_red = np.array([140,   10, 240]) 
upper_red = np.array([180, 130, 256])
RES_WIDTH = 1920
RES_HEIGHT = 1080

with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
    loaded_dict = pickle.load(f)
    laser_points = loaded_dict["laser_points"]
    camera_points = loaded_dict["camera_points"]
   
laser_points = laser_points[:, :3*CALIBRATION_ITER]
camera_points = camera_points[:, :3*CALIBRATION_ITER]
R, t = optimal_rotation_and_translation(camera_points, laser_points)



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
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)



    cv2.namedWindow('Laser Detector',cv2.WINDOW_NORMAL)


    a = np.linspace(int(3*RES_WIDTH/16), int(13*RES_WIDTH/16), sample_x)
    b = np.linspace(int(2*RES_HEIGHT/8), int(5*RES_HEIGHT/8), sample_y)

    x = np.tile(a, sample_y)
    y = np.repeat(b, sample_x)
 
    rmse_scores = []
    mouse3d_pos = []

    for i in range(len(x)):        
        
        
        
        capture = device.update()        
        ret_color, color_image = capture.get_color_image()        
        ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        while not ret_color or not ret_depth:
            capture = device.update()        
            ret_color, color_image = capture.get_color_image()        
            ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

        
        pix_x = int(x[i])
        pix_y = int(y[i])
        rgb_depth = transformed_depth_image[pix_y, pix_x]
        pixels = k4a_float2_t((pix_x, pix_y))

        pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)  

        mouse_in_camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))     
        mouse3d_pos.append(mouse_in_camera_coordinates.reshape((-1)) )   
        # rotate and translate
        camera_coordinates_in_laser_coordinates =  R @ mouse_in_camera_coordinates + t        
        coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)
        y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1], camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis
                
        if(len(y_m) > 0 and len(x_m) > 0):
            si_0.SetXY(y_m[0])        
            si_1.SetXY(x_m[0]) 



    
        time.sleep(0.5)
        average_laser_pos_in_camera_coordinates_np = 0
        number_of_camera_coordinates_in_batch = 0
        while  number_of_camera_coordinates_in_batch < num_iterations:
            capture = device.update()
            ret_color, color_image = capture.get_color_image()
            #ret_depth, transformed_depth_image = capture.get_transformed_depth_image()

            if not ret_color:
                continue

            circle = detect_circle_position(color_image, lower_range=lower_red, upper_range=upper_red)

            if circle is  None:
                print("Laser is not detected")                
                continue

            circle = np.round(circle).astype("int")
            cv2.circle(color_image, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)
            cv2.circle(color_image, center=(pix_x, pix_y), radius=circle[2], color=(255, 0, 0), thickness=2)

            pix_x = circle[0]
            pix_y = circle[1]
            rgb_depth = transformed_depth_image[pix_y, pix_x]

            pixels = k4a_float2_t((pix_x, pix_y))

            pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
            laser_in_camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z])

        
            cv2.imshow('Laser Detector', color_image)
            average_laser_pos_in_camera_coordinates_np += laser_in_camera_coordinates
            number_of_camera_coordinates_in_batch += 1

    


    
        average_laser_pos_in_camera_coordinates_np /= number_of_camera_coordinates_in_batch


        rmse = np.sqrt(np.mean(np.square(average_laser_pos_in_camera_coordinates_np - mouse_in_camera_coordinates.reshape((-1)))))
        print("RMSE: ", rmse)
        rmse_scores.append(rmse)

      
        # cv2.circle(color_image, center=(mouse_x, mouse_y), radius=10, color=(0, 255, 0), thickness=2)
        # Show detected target position
        cv2.imshow('Laser Detector',color_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break


    
    accuracy_results = {"pointer_pos": np.array(mouse3d_pos),
                        "rmse": rmse_scores}

    with open('{}/iter_{}.pkl'.format(accuracy_path, CALIBRATION_ITER), 'wb') as f:
        pickle.dump(accuracy_results, f)

    mre2.disconnect()
    print("done")



if __name__ == "__main__":
     main()