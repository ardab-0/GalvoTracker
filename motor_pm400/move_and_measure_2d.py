from motor_and_power_meter_controller import MotorAndPowerMeterController
from utils import interpolate_and_lowpass
import numpy as np

controller = MotorAndPowerMeterController()

(port_x, unpacker_x, port_y, unpacker_y) = controller.initializeMotors("COM5", "COM4")
print("Motors initialized and homed.")
controller.initializePM400()
print("PM400 initialized.")

# measure x axis
measurement_dictinary = controller.moveMotorAndMeasure(port_x, unpacker_x, 30)

# Filter the measurement and find peak to detect laser position
t_filtered, measurements_filtered_mw = interpolate_and_lowpass(measurement_dictinary["t_s"], measurement_dictinary["measurements_mw"])
distances_filtered_mm = t_filtered * measurement_dictinary["speed_mms"]
maximum_pos_filtered_mm = distances_filtered_mm[np.argmax(measurements_filtered_mw)]


# go to peak power position to get y axis measurement
controller.moveMotor(port_x, unpacker_x, maximum_pos_filtered_mm )


measurement_dictinary = controller.moveMotor(port_y, unpacker_y, 8)

port_x.close()
port_y.close()
controller.closePM400()