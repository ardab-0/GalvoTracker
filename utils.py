    
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
	tnew = np.arange(0.05, t[-1], T)
	ycubic = fcubic(tnew)
	return tnew, ycubic


def moving_average(x, N):
	return np.convolve(x, np.ones(N)/N, mode='same')



def apply_filter(x, h):
	return np.convolve(x, h, "same")



def optimal_rotation_and_translation(A, B):
	"""
		A: points (3xN)
		B: points (3xN)

		source: https://nghiaho.com/?page_id=671
				https://en.wikipedia.org/wiki/Kabsch_algorithm
	"""

	centroidA = np.mean(A, axis=1).reshape((-1, 1))
	centroidB = np.mean(B, axis=1).reshape((-1, 1))
	# print(centroidA)
	# print(centroidB)
	H = (A - centroidA) @ (B - centroidB).T
	
	U, S, Vh = np.linalg.svd(H)

	V = Vh.T
	# print(U.shape)
	# print(S.shape)
	# print(V.shape)

	R = V @ U.T

	if np.linalg.det(R) < 0:
		print("negative det")
		V[:, 2] *= -1
		R = V @ U.T
	
	t = centroidB - R @ centroidA


	return R , t




# A = np.array([	[1, 2, 3, 4],
# 				[0, 0, 0, 0],
# 				[0, 0, 0, 0] ])


# a = 90 / 180 * np.pi
# rot = np.array([[np.cos(a), -np.sin(a), 0],
# 				[np.sin(a), np.cos(a), 0],
# 				[0, 0, 1] ])


# B = rot @ A + np.array([10.2, 0.08, 5]).reshape((-1, 1))

# print(B)



# R , t = optimal_rotation_and_translation(A, B)

# print(R)
# print(t)