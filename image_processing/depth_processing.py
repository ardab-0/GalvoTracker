import cv2
import pykinect_azure as pykinect
from pykinect_azure import K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH, k4a_float2_t
import numpy as np
import time
from PIL import Image



def main():            
   

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



    cv2.namedWindow('Depth Image',cv2.WINDOW_NORMAL)

    i = 0
    while True:
        start = time.time()
        # Get capture
        
        capture = device.update()
        print("Time until capture (s): ", time.time() - start)

        # # Get the color image from the capture
        ret_color, color_image = capture.get_color_image()
        print("Time until color image (s): ", time.time() - start)

        # Get the colored depth
        ret_colored_depth, colored_depth_image = capture.get_transformed_colored_depth_image()
        print("Time until colored depth image (s): ", time.time() - start)
        
        # Get the colored depth
        ret_depth, depth_image = capture.get_transformed_depth_image()
        print("Time until depth image (s): ", time.time() - start)
        
        if not ret_color or not ret_colored_depth or not ret_depth:
            continue
        
        im_color = Image.fromarray(color_image)
        im_color.save(f"image_processing/sphere_ims/color_{i}.png")
        im_colored_depth = Image.fromarray(colored_depth_image)
        im_colored_depth.save(f"image_processing/sphere_ims/colored_depth_image_{i}.png")
        np.save(f"image_processing/sphere_ims/depth_image_{i}.npy", depth_image)
        
        i+=1
        # pix_x = mouse_x
        # pix_y = mouse_y
        # rgb_depth = transformed_depth_image[pix_y, pix_x]

        # pixels = k4a_float2_t((pix_x, pix_y))



        # pos3d_color = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR)
        # # pos3d_depth = device.calibration.convert_2d_to_3d(pixels, rgb_depth, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_DEPTH)
        # # print(f"RGB depth: {rgb_depth}, RGB pos3D: {pos3d_color}, Depth pos3D: {pos3d_depth}")

        # camera_coordinates = np.array([pos3d_color.xyz.x, pos3d_color.xyz.y, pos3d_color.xyz.z]).reshape((3, 1))

    
        # # Test point cloud 
        # # point_cloud_image = transformation.depth_image_to_point_cloud(transformed_depth_image, K4A_CALIBRATION_TYPE_COLOR)
        

        
        # # rotate and translate

        # camera_coordinates_in_laser_coordinates =  R @ camera_coordinates + t


        # # camera_coordinates_in_laser_coordinates = np.floor(camera_coordinates_in_laser_coordinates)

        # # print("camera_coordinates", camera_coordinates)

        # # print("camera_coordinates_in_laser_coordinates", camera_coordinates_in_laser_coordinates)


        # coordinate_transform = CoordinateTransform(d=d, D=camera_coordinates_in_laser_coordinates[2], rotation_degree=mirror_rotation_deg)



        # y_m, x_m = coordinate_transform.target_to_mirror(camera_coordinates_in_laser_coordinates[1], camera_coordinates_in_laser_coordinates[0]) # order is changed in order to change x and y axis
        # print("Time until completing calculations (s): ", time.time() - start)
        
        
        # if(len(y_m) > 0 and len(x_m) > 0):
        #     si_0.SetXY(y_m[0])        
        #     si_1.SetXY(x_m[0]) 


        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(color_image, f"fps: {1 / (time.time() - start)}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # cv2.putText(color_image, f"Target Coordinates w.r.t. mirror center:", (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"X: {camera_coordinates_in_laser_coordinates[0]}", (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"Y: {camera_coordinates_in_laser_coordinates[1]}", (10, 80), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(color_image, f"Z: {camera_coordinates_in_laser_coordinates[2]}", (10, 100), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


        
        # cv2.circle(color_image, center=(mouse_x, mouse_y), radius=10, color=(0, 255, 0), thickness=2)
        # Show detected target position
        cv2.imshow('Depth Image',colored_depth_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break


    print("done")



if __name__ == "__main__":
     main()