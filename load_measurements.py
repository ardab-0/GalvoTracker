import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np


distance_to_plane = "440mm"
distance_to_mirror_center = "d0"


def compare_measurements(n):
    for i in range(n):
        with open('measurements/{}/{}/{}.pkl'.format(distance_to_mirror_center, distance_to_plane, str(-1*i)), 'rb') as f:
            loaded_dict = pickle.load(f)
            print(str(-1*i) + ":   " + str(loaded_dict["maximum_pos_mm"]))


# compare_measurements(6)

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


def plot_measurements(filename, distance_to_mirror_center, distance_to_plane, position):
    with open('{}/{}/{}/{}.pkl'.format(filename, distance_to_mirror_center, distance_to_plane, position), 'rb') as f:
        loaded_dict = pickle.load(f)
        times = loaded_dict["t_s"]
        measurements_mW = loaded_dict["measurements_mw"]
        distances_mm = loaded_dict["distances_mm"]
        maximum_pos_mm = loaded_dict["maximum_pos_mm"]      
        
        
        
        
    # Interpolation
    T = 0.01
    N = 100
    times_reg, measurements_reg_mW = interpolate_and_lowpass(times, measurements_mW, T=T, N=N)
    total_time = times[-1]
    speed = 30 / total_time
    distances_reg_mm = times_reg * speed
    maximum_pos_reg_mm = distances_reg_mm[np.argmax(measurements_reg_mW)]

    ##  Plotting
    plt.subplot(2, 2, 1)
    plt.plot(times, measurements_mW)
    plt.ylabel("Power (mW)")
    plt.xlabel("t (s)")
    plt.title("Power-Time (Raw Sensor Data)")

    plt.subplot(2,2,2)
    plt.plot(times_reg, measurements_reg_mW)
    plt.ylabel("Power (mW)")
    plt.xlabel("time (s)")
    plt.title("Power-Time (Cubic Interpolation Period {} + Moving Average Filter Length:{})".format(T, N))



    plt.subplot(2, 2, 3)
    plt.plot(distances_reg_mm, measurements_reg_mW)
    plt.ylabel("Power (mW)")
    plt.xlabel("distance (mm)")
    plt.title("Power-Distance (Cubic Interpolation Period {} + Moving Average Filter Length:{})".format(T, N))


    ax = plt.subplot(2,2, 4)
     

    ax.text(0.1, 0.8, 'Maximum Position (mm): {}'.format(maximum_pos_mm))
    ax.text(0.1, 0.6, 'Maximum Position After Interpolation + MA (mm): {}'.format(maximum_pos_reg_mm))  

    plt.show()




plot_measurements("measurements_2","d0", "410mm", "0x0_0")