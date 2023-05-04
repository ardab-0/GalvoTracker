from motor_and_power_meter_controller import MotorAndPowerMeterController
from utils import interpolate_and_lowpass
import numpy as np
import matplotlib.pyplot as plt

# Before running the script unplug and plug pm400, it might get stuck 

def to_abs(initial_pos_mm, pos_mm):
    """
    Convert relative position to absolute position with respect to initial position
    """
    return initial_pos_mm + pos_mm


T=0.01
N=100

controller = MotorAndPowerMeterController()

controller.initializeMotors("COM5", "COM4")
print("Motors initialized and homed.")
controller.initializePM400()
print("PM400 initialized.")

initial_x_pos_mm = 20
initial_y_pos_mm = 9

scanline_x_mm =  10 
scanline_y_mm =  10

# move motors to initial positions
controller.moveMotorsAbsolute(initial_x_pos_mm, initial_y_pos_mm)
print("reached initial position.")


# measure x axis
measurement_dictionary = controller.moveMotorRelativeAndMeasure(scanline_x_mm, motor_id="x")

# Filter the measurement and find peak to detect laser position
t_filtered, measurements_filtered_mw = interpolate_and_lowpass(measurement_dictionary["t_s"], measurement_dictionary["measurements_mw"], T=T, N=N)
distances_filtered_mm = t_filtered * measurement_dictionary["speed_mms"]
maximum_pos_filtered_mm = distances_filtered_mm[np.argmax(measurements_filtered_mw)]  # relative distance with respect to initial position 

##  Plotting
plt.subplot(2, 2, 1)
plt.plot(measurement_dictionary["t_s"], measurement_dictionary["measurements_mw"])
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



# go to peak power position to get y axis measurement
controller.moveMotorAbsolute(to_abs(initial_x_pos_mm, maximum_pos_filtered_mm) , motor_id="x")

input("wait...")
measurement_dictinary = controller.moveMotorAbsolute(to_abs(initial_y_pos_mm, 8), motor_id="y")

controller.close_motors()
controller.closePM400()