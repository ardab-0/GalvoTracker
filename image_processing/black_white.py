import cv2
import numpy as np
from image_processing.circle_detector import detect_circle_position





def black_and_white_threshold(im):
	image = np.copy(im)
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image_gray = cv2.resize(image_gray, (1280, 720), interpolation=cv2.INTER_AREA)
	thresh = 200
	im_bw = cv2.threshold(image_gray, thresh, 255, cv2.THRESH_BINARY)[1]
	image[im_bw < 10 ] = 0
	return image



def test_black_and_white_threshold():
	lower_red = np.array([155,   10, 240]) 
	upper_red = np.array([165, 30,256])

	image = cv2.imread("test_images/target0.jpg")
	image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
	cv2.imshow("image_color", image)

	image = black_and_white_threshold(image)
	circle = detect_circle_position(image, lower_range=lower_red, upper_range=upper_red)

	if circle is  None:
		print("Laser is not detected")                
		

	circle = np.round(circle).astype("int")
	cv2.circle(image, center=(circle[0], circle[1]), radius=circle[2], color=(0, 255, 0), thickness=2)


	cv2.namedWindow("image")


	while True:
		
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF	
		
		if key == ord("q"):
			break

	# close all open windows
	cv2.destroyAllWindows()