import cv2
import numpy as np




def click(event, x, y, flags, param):
	
	if event == cv2.EVENT_LBUTTONDOWN:
		print(param[y, x])
	

i = 5

image = np.load(f"image_processing/sphere_ims/depth_image_{i}.npy")
image = image / np.max(image)

image = cv2.imread(f"image_processing/sphere_ims/colored_depth_image_{i}.png")


cv2.namedWindow("image")
cv2.setMouseCallback("image", click, image)

while True:
	
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF	
	
	if key == ord("q"):
		break

# close all open windows
cv2.destroyAllWindows()