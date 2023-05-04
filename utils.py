    
from scipy import interpolate
import numpy as np



def interpolate_and_lowpass(t, x, T, N):
        """
        t: irregular time vector
        x: irregular measurement vector
        T: interpolation period
        N: moving average filter length
        """
        fcubic = interpolate.interp1d(t, x, kind='cubic')
        tnew = np.arange(0.005, t[-1], T)
        ycubic = fcubic(tnew)

        smooth_x = np.convolve(ycubic, np.ones(N)/N, mode='same')
        return tnew, smooth_x