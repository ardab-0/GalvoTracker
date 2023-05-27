import cv2






image = cv2.imread("test_images/target0.jpg")
image = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_AREA)
iamge_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
iamge_gray = cv2.resize(iamge_gray, (1280, 720), interpolation=cv2.INTER_AREA)
thresh = 220
im_bw = cv2.threshold(iamge_gray, thresh, 255, cv2.THRESH_BINARY)[1]
image[im_bw < 10 ] = 0

cv2.namedWindow("image")


while True:
	
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF	
	
	if key == ord("q"):
		break

# close all open windows
cv2.destroyAllWindows()