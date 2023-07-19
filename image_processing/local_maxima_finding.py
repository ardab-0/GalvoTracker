from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
import cv2



def find_local_maxima(im, number_of_maxima):
    """
    im: grayscale float image
    returns largest "number_of_maxima" local maxima coordinates
    """  
    
    # Comparison between image_max and im to find the coordinates of local maxima
    # coordinates are sorted from largest to smallest value
    coordinates = peak_local_max(im, min_distance=20)
    if len(coordinates) < number_of_maxima:
        return coordinates
    else:
        return coordinates[:number_of_maxima]

    

    



def test_find_local_maxima():
    im = cv2.imread("image_processing/gauss.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = img_as_float(im)

    coords = find_local_maxima(im, 2)

    # display results
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    # ax[1].imshow(image_max, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('Maximum filter')

    ax[1].imshow(im, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(coords[:, 1], coords[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    fig.tight_layout()

    plt.show()




if __name__ == "__main__":
    test_find_local_maxima()