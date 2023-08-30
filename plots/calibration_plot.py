import matplotlib.pyplot as plt


z = [2, 4, 6, 8, 10]

e = [2.27, 2.10, 1.90, 1.08, 1.0]

plt.plot(z, e)
plt.grid()
plt.xlabel("Calibration points")
plt.ylabel("Absolute calibration error (mm)")
plt.show()