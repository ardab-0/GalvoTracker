import numpy as np
import matplotlib.pyplot as plt



def moving_average(x, N):
	return np.convolve(x, np.ones(N)/N, mode='same')

turning_point = 30


# y = np.where(x<turning_point, x, -x+2*turning_point)
t = np.linspace(0, 10, 300)
x = 10 * t + 1/2 * 4 *t**2

x_n = x + np.random.random(x.shape) * 1
# y_n = y + np.random.random(y.shape) * 1

x_f = moving_average(x_n, 5)

dt =  t[1:] - t[:-1]

v = (x_n[1:] - x_n[:-1]) / dt
v_f = moving_average(v, 10)

a  = (v_f[1:] - v_f[:-1]) / dt[1:]
a_f = moving_average(a, 100)


plt.subplot(1, 6, 1)
plt.plot(t, x_n)
plt.xlabel("t")
plt.ylabel("x_n")


plt.subplot(1, 6, 2)
plt.plot(t, x_f)
plt.xlabel("t")
plt.ylabel("x_f")

plt.subplot(1, 6, 3)
plt.plot(t[1:], v)
plt.xlabel("t")
plt.ylabel("v")

plt.subplot(1, 6, 4)
plt.plot(t[1:], v_f)
plt.xlabel("t")
plt.ylabel("v_f")


plt.subplot(1, 6, 5)
plt.plot(t[2:], a_f)
plt.xlabel("t")
plt.ylabel("a_f")

plt.show()

