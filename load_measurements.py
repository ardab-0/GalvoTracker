import pickle
import matplotlib.pyplot as plt
import numpy as np
from motor_pm400.utils import interpolate_and_lowpass





def plot_measurements(filename, distance_to_mirror_center, distance_to_plane, position, T, N):
    """
        T = interpolation period
        N = Moving average filter length
    """
    
    
    with open('{}/{}/{}/{}.pkl'.format(filename, distance_to_plane, distance_to_mirror_center, position), 'rb') as f:
        loaded_dict = pickle.load(f)
        times = loaded_dict["t_s"]
        measurements_mW = loaded_dict["measurements_mw"]
        distances_mm = loaded_dict["distances_mm"]
        speed_mms = loaded_dict["speed_mms"]      
        
        
        
        
    
    times_reg, measurements_reg_mW = interpolate_and_lowpass(times, measurements_mW, T=T, N=N)
    
    distances_reg_mm = times_reg * speed_mms
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
     

    ax.text(0.1, 0.6, 'Maximum Position After Interpolation \n + MA (mm): {}'.format(maximum_pos_reg_mm))  

    plt.show()




plot_measurements("measurements_vertical","d0", "210mm", "0.0x-3.0", T=0.01, N=100)