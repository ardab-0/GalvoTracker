import cv2
import numpy as np



class Color_Picker():

	def __init__(self, image):
		self.image = image
		self.color = None


	def click(self, event, x, y, flags, param):
		
		if event == cv2.EVENT_LBUTTONDOWN:
			print("Chosen color: ", self.image[y, x])
			print("Press q to exit.")
			self.color = self.image[y, x]
		

	# i = 5

	# image = np.load(f"image_processing/sphere_ims/depth_image_{i}.npy")
	# image = image / np.max(image)

	# image = cv2.imread(f"image_processing/sphere_ims/colored_depth_image_{i}.png")

	def choose_pixel(self):
		cv2.namedWindow("image")
		cv2.setMouseCallback("image", self.click)

		while True:
			
			cv2.imshow("image", self.image)
			key = cv2.waitKey(1) & 0xFF	
			
			if key == ord("q"):
				break

		# close all open windows
		cv2.destroyAllWindows()
		return self.color


if __name__ == "__main__":
	i = 5
	image = cv2.imread(f"image_processing/sphere_ims/colored_depth_image_{i}.png")
	cp = Color_Picker(image)

	print(cp.choose_pixel())