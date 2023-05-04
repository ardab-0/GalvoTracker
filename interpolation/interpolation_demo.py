import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

x = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
y = np.exp(-x/3.0)
flinear = interpolate.interp1d(x, y)
fcubic = interpolate.interp1d(x, y, kind='cubic')

xnew = np.arange(0.001, 20, 1)
ylinear = flinear(xnew)
ycubic = fcubic(xnew)
plt.plot(x, y, 'X', xnew, ylinear, 'x', xnew, ycubic, 'o')
plt.show()