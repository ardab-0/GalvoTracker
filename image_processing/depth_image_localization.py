import cv2
import numpy as np
import matplotlib.pyplot as plt


i = 5

image = np.load(f"image_processing/sphere_ims/depth_image_{i}.npy")
# image = image / np.max(image) * 255
# image = image.astype("uint8")
# image = cv2.imread(f"image_processing/sphere_ims/colored_depth_image_{i}.png")


cv2.namedWindow("image")

img_median = cv2.medianBlur(image, 5) 

cv2.imshow('img_median', img_median) 
cv2.imshow('image', image) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
	


sobelx = cv2.Sobel(img_median,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img_median,cv2.CV_64F,0,1,ksize=5)

sobelxx = cv2.Sobel(sobelx,cv2.CV_64F,1,0,ksize=5)
sobelyy = cv2.Sobel(sobely,cv2.CV_64F,0,1,ksize=5)


w = sobelxx - sobelyy
w_shape = w.shape
w = w.reshape(-1)


argmax = np.flip(np.argsort(w))
w = w.reshape(w_shape)

for i in range(10):
    row = argmax[i] // w_shape[1]
    col = argmax[i] % w_shape[1]
    print(w[row, col])
    plt.scatter( col , row , s = 70, color="r" )


plt.imshow(w)

plt.title("VORTICITY")
plt.show()

# sobelx = sobelx[::10, ::10]
# sobely = sobely[::10, ::10]

# # max = np.max([np.max(sobelx), np.max(sobely)]) 
# # min = np.min([np.min(sobelx), np.min(sobely)]) 

# # normalization_coef = np.max([np.abs(max), np.abs(min)])

# mag = np.sqrt(sobelx**2 + sobely**2)
# sobelx /= mag 
# sobely /= mag 

# sobelx = np.nan_to_num(sobelx)
# sobely = np.nan_to_num(sobely)




# cv2.imshow('sobel x', sobelx) 
# cv2.imshow('sobel y', sobely) 

# cv2.waitKey(0)
# cv2.destroyAllWindows() 


# x,y = np.meshgrid(np.linspace(1,1280,128),np.linspace(1,720,72))



# plt.quiver(x,y,np.flip(sobelx, axis=0),np.flip(sobely, axis=0))
# plt.show()