import cv2

import pykinect_azure as pykinect

if __name__ == "__main__":

	# Initialize the library, if the library is not found, add the library path as argument
	pykinect.initialize_libraries()

	# Modify camera configuration
	device_config = pykinect.default_configuration
	device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
	# print(device_config)

	# Start device
	device = pykinect.start_device(config=device_config)

	cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)

	image_index = 0
	while True:

		# Get capture
		capture = device.update()

		# Get the color image from the capture
		ret, color_image = capture.get_color_image()

		if not ret:
			continue
			
		# Plot the imageq
		cv2.imshow("Color Image",color_image)
		if cv2.waitKey(1) == ord('s'):
			cv2.imwrite('test_images/target{}.jpg'.format(image_index), color_image)
			image_index += 1

		# Press q key to stop
		if cv2.waitKey(1) == ord('q'): 
			break
