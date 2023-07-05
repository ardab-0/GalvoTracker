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
        
        cv2.imshow('Depth Image',colored_depth_image)
        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            break

    print("done")



if __name__ == "__main__":
     main()