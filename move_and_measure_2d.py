from motor_and_power_meter_controller import MotorAndPowerMeterController



controller = MotorAndPowerMeterController()

(port, unpacker) = controller.initializeMotor("COM10")
controller.initializePM400()
measurement_dictinary = controller.moveMotorAndMeasure(port, unpacker, 8)

print(measurement_dictinary)



controller.moveMotor(port, unpacker, measurement_dictinary["maximum_pos_mm"] )
