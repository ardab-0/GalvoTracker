from motor_and_power_meter_controller import MotorAndPowerMeterController
from utils import interpolate_and_lowpass


controller = MotorAndPowerMeterController()

(port_x, unpacker_x, port_y, unpacker_y) = controller.initializeMotors("COM5", "COM3")
controller.initializePM400()


# measure x axis
measurement_dictinary = controller.moveMotorAndMeasure(port_x, unpacker_x, 8)


t_filtered, measurements_filtered_mw = interpolate_and_lowpass(measurement_dictinary["t_s"], measurement_dictinary["measurements_mw"])


# go to peak power position to get y axis measurement
controller.moveMotor(port_x, unpacker_x, measurement_dictinary["maximum_pos_mm"] )


measurement_dictinary = controller.moveMotor(port_y, unpacker_y, 8)

port_x.close()
port_y.close()
controller.closePM400()