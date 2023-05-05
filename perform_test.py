from motor_pm400.motor_and_power_meter_controller import MotorAndPowerMeterController
from motor_pm400.utils import interpolate_and_lowpass
import numpy as np
import matplotlib.pyplot as plt
import optoMDC
import time
from mirror.coordinate_transformation import CoordinateTransform
import pickle
import os

# Before running the script unplug and plug pm400, it might get stuck 


# Constants

T=0.01 # interpolation period
N=100 # moving average filter length
d = 0 # distance from mirror surface to rotation center
D = 410 # distance from target plane to mirror
mirror_rotation_deg = 45 # mirror rotation degree

initial_x_pos_mm = 20 # initial position of motor x
initial_y_pos_mm = 9 # initial position of motor y
scanline_x_mm =  10 # length of scan line in x direction
scanline_y_mm =  10 # length of scan line in y direction

measurement_foldername = "measurements_3"


# Functions

def to_abs(initial_pos_mm, pos_mm):
    """
    Convert relative position to absolute position with respect to initial position
    """
    return initial_pos_mm + pos_mm


def plot(t, measurements_mw, t_filtered, measurements_filtered_mw, distances_filtered_mm, maximum_pos_filtered_mm):
    ##  Plotting
    plt.subplot(2, 2, 1)
    plt.plot(t, measurements_mw)
    plt.ylabel("Power (mW)")
    plt.xlabel("t (s)")
    plt.title("Power-Time (Raw Sensor Data)")

    plt.subplot(2,2,2)
    plt.plot(t_filtered, measurements_filtered_mw)
    plt.ylabel("Power (mW)")
    plt.xlabel("time (s)")
    plt.title("Power-Time (Cubic Interpolation Period {} + Moving Average Filter Length:{})".format(T, N))



    plt.subplot(2, 2, 3)
    plt.plot(distances_filtered_mm, measurements_filtered_mw)
    plt.ylabel("Power (mW)")
    plt.xlabel("distance (mm)")
    plt.title("Power-Distance (Cubic Interpolation Period {} + Moving Average Filter Length:{})".format(T, N))


    ax = plt.subplot(2,2, 4)
        

    ax.text(0.1, 0.6, 'Maximum Position After Interpolation + MA (mm): {}'.format(maximum_pos_filtered_mm))  

    plt.show()


def create_folder_structure(path): 
    if not os.path.exists(path):   
        os.makedirs(path)




# create necessary folder structur if necessary
distance_to_plane = "{}mm".format(D) 
distance_to_mirror_center = "d{}".format(d)
save_path = "{}/{}/{}/".format(measurement_foldername, distance_to_plane, distance_to_mirror_center)
create_folder_structure(save_path)

# initialize mirrors
mre2 = optoMDC.connect()
mre2.reset()

# Set up mirror in closed loop control mode(XY)
ch_0 = mre2.Mirror.Channel_0
ch_0.StaticInput.SetAsInput()                       # (1) here we tell the Manager that we will use a static input
ch_0.SetControlMode(optoMDC.Units.XY)           
ch_0.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_0 = mre2.Mirror.Channel_0.StaticInput


ch_1 = mre2.Mirror.Channel_1
ch_1.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
ch_1.SetControlMode(optoMDC.Units.XY)           
ch_1.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_1 = mre2.Mirror.Channel_1.StaticInput

coordinate_transform = CoordinateTransform(d=d, D=D, rotation_degree=mirror_rotation_deg)



controller = MotorAndPowerMeterController()

controller.initializeMotors("COM5", "COM4")
print("Motors initialized and homed.")
controller.initializePM400()
print("PM400 initialized.")



# move motors to initial positions
controller.moveMotorsAbsolute(initial_x_pos_mm, initial_y_pos_mm)
print("reached initial position.")


# perform measurements for given target plane positions and save them
# might need to change order
x_t = np.array([-3, -2, -1, 0, 1, 2, 3])
y_t = np.array([0])
x_m, y_m = coordinate_transform.target_to_mirror(x_t, y_t)

# Set mirror position
for i in range(len(x_t)):
    for j in range(len(y_t)):

        si_0.SetXY(x_m[i])        
        si_1.SetXY(y_m[j])  
        time.sleep(0.1)

        # measure x axis
        measurement_dictionary = controller.moveMotorRelativeAndMeasure(scanline_x_mm, motor_id="x")


        position_input_mm = "{}x{}".format(x_t[i], y_t[j])

        with open('{}{}.pkl'.format(save_path, position_input_mm), 'wb') as f:
            pickle.dump(measurement_dictionary, f)
        
        # return to initial position for next measurement
        controller.moveMotorAbsolute(initial_x_pos_mm , motor_id="x")




# # Filter the measurement and find peak to detect laser position
# t_filtered, measurements_filtered_mw = interpolate_and_lowpass(measurement_dictionary["t_s"], measurement_dictionary["measurements_mw"], T=T, N=N)
# distances_filtered_mm = t_filtered * measurement_dictionary["speed_mms"]
# maximum_pos_filtered_mm = distances_filtered_mm[np.argmax(measurements_filtered_mw)]  # relative distance with respect to initial position 



# plot(measurement_dictionary["t_s"], measurement_dictionary["measurements_mw"], t_filtered, measurements_filtered_mw, distances_filtered_mm, maximum_pos_filtered_mm)



# Closing operations
controller.close_motors()
controller.closePM400()

si_0.SetXY(0)        
si_1.SetXY(0)
mre2.disconnect()
print("done")
