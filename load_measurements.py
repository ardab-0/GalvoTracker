import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import interpolate_and_lowpass


def print_difference(filename, distance_to_plane, distance_to_mirror_center, T, N, measurement_count, x_pos=None, y_pos=None):
    result_dict = {}

    for i in range(-int(measurement_count/2), int(measurement_count/2)+1 ):
        if x_pos and y_pos is None:
            position = "{}x{}".format(x_pos, float(i/2) )
        elif y_pos and x_pos is None:
            position = "{}x{}".format(float(i/2), y_pos )
        else:
            print("Enter only one of x_pos and y_pos")
            return
                
        with open('{}/{}/{}/{}.pkl'.format(filename, distance_to_plane, distance_to_mirror_center, position), 'rb') as f:
            loaded_dict = pickle.load(f)
            times = loaded_dict["t_s"]
            measurements_mW = loaded_dict["measurements_mw"]
            distances_mm = loaded_dict["distances_mm"]
            speed_mms = loaded_dict["speed_mms"] 

        times_reg, measurements_reg_mW = interpolate_and_lowpass(times, measurements_mW, T=T, N=N)
    
        distances_reg_mm = times_reg * speed_mms
        maximum_pos_reg_mm = distances_reg_mm[np.argmax(measurements_reg_mW)]
        result_dict[str(float(i/2))] = maximum_pos_reg_mm
    
    print("Peak Power Position (mm)\n")
    for key in result_dict:
        print(round(result_dict[key], 3))

    print("\n\n")
    print("Relative Distance(mm) \n")

    reference = result_dict["0.0"]
    for key in result_dict:
        print(round(result_dict[key] - reference, 3))

def plot_measurements(filename, distance_to_plane, distance_to_mirror_center, position, T, N):
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

# plot_measurements("power_measurements/measurements_horizontal", "425mm", "d0", "3.0x-5.0", T=0.01, N=100)



print_difference("power_measurements/measurements_horizontal", "425mm", "d0", T=0.01, N=100, measurement_count=13, y_pos="5.0")