    
from scipy import interpolate
import numpy as np



def interpolate_and_lowpass(t, x, T, N):
	"""
	t: irregular time vector
	x: irregular measurement vector
	T: interpolation period
	N: moving average filter length
	"""
	tnew, ycubic = interpolate_cubic(t, x, T)

	smooth_x = moving_average(ycubic, N)
	return tnew, smooth_x


def interpolate_cubic(t, x, T):

	fcubic = interpolate.interp1d(t, x, kind='cubic')
	tnew = np.arange(0.005, t[-1], T)
	ycubic = fcubic(tnew)
	return tnew, ycubic


def moving_average(x, N):
	return np.convolve(x, np.ones(N)/N, mode='same')



def apply_filter(x, h):
	return np.convolve(x, h, "same")