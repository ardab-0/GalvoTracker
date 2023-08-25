"""
Source: https://stackoverflow.com/questions/28281742/fitting-a-circle-to-a-binary-image
"""

from skimage import io, color, measure, draw, img_as_bool
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def fit_circle(image):
   
    image = img_as_bool(image[..., 0])
    regions = measure.regionprops(measure.label(image))
    bubble = regions[0]

    y0, x0 = bubble.centroid
    r = bubble.major_axis_length / 2.

    def cost(params):
        x0, y0, r = params
        coords = draw.disk((y0, x0), r, shape=image.shape)
        template = np.zeros_like(image)
        template[coords] = 1
        return -np.sum(template == image)

    x0, y0, r = optimize.fmin(cost, (x0, y0, r))

    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    circle = plt.Circle((x0, y0), r)
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.add_artist(circle)
    print(f"X: {x0}, Y: {y0}")
    plt.show()

    return x0, y0


def test_fit_circle():
    image = io.imread('image_processing/laser3.png')
    fit_circle(image)