import cv2





def click(event, x, y, flags, param):
	
	if event == cv2.EVENT_LBUTTONDOWN:
		print(param[y, x])
	



image = cv2.imread("test_images/laser_im1.jpg")
image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
# Convert BGR to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)



cv2.namedWindow("image")
cv2.setMouseCallback("image", click, image_hsv)

while True:
	
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF	
	
	if key == ord("q"):
		break

# close all open windows
cv2.destroyAllWindows()