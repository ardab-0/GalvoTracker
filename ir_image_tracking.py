import pykinect_azure as pykinect
import cv2
from pykinect_azure.k4a.transformation import Transformation
from pykinect_azure import Image
from pykinect_azure.k4a import _k4a
import time
from circle_detector_library.circle_detector_module import *


DOWNSAMPLE = 4



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

transformation = Transformation(device.get_calibration(
    device_config.depth_mode, device_config.color_resolution))


cv2.namedWindow('image', cv2.WINDOW_NORMAL)


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
    transformed_ir_image[transformed_ir_image>5000] = 5000
    transformed_ir_image = transformed_ir_image / 5000 * 255
    transformed_ir_image = transformed_ir_image.astype("uint8")
    transformed_ir_image = cv2.cvtColor(transformed_ir_image,cv2.COLOR_GRAY2RGB)
    print("Time until transformed ir image (s): ", time.time() - start)

    new_circle = circle_detector.detect_np(transformed_ir_image, prevCircle)    
    prevCircle = new_circle
    transformed_ir_image = cv2.circle(transformed_ir_image, (int(new_circle.x), int(new_circle.y)), radius=10, color=(0, 255, 0), thickness=2)


    cv2.imshow('image',transformed_ir_image)
    # Press q key to stop
    if cv2.waitKey(1) == ord('q'):
        break



    
