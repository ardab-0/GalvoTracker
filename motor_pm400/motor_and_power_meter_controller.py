"""
Created on Thu Apr 20 16:06:10 2023

@author: ki52soho
"""

import thorlabs_apt_protocol as apt
import time
import serial
import keyboard
import usbtmc
import numpy as np


COM_PORT_X = "COM10"
COM_PORT_Y = "COM9"

class MotorAndPowerMeterController:
    def __init__(self):    
        self.MM_TO_ENCODER = 25000 # from Z606 motorized actuator documentation

        # from endpoint enums in thorlabs_apt_device library
        self.HOST = 0x01
        self.RACK = 0x11
        self.BAY0 = 0x21
        self.BAY1 = 0x22
        self.BAY2 = 0x23
        self.BAY3 = 0x24
        self.BAY4 = 0x25
        self.BAY5 = 0x26
        self.BAY6 = 0x27
        self.BAY7 = 0x28
        self.BAY8 = 0x29
        self.BAY9 = 0x2A
        self.USB = 0x50

        self.CHANNEL = 1 # first channel in controller (there is one channel in tdc001)
        # from endpoint enums in thorlabs_apt_device library


        self.portX = None
        self.unpackerX = None

        self.portY = None
        self.unpackerY = None


    def initializeMotor(self, com_port):
        port = serial.Serial(com_port, 115200, rtscts=True, timeout=0.01)
        port.rts = True
        port.reset_input_buffer()
        port.reset_output_buffer()
        port.rts = False
        port.write(apt.mot_move_home(source=self.HOST, dest=self.BAY0 ,chan_ident=self.CHANNEL))
        unpacker = apt.Unpacker(port)

        is_homed = False
        while(not is_homed):    
            if keyboard.is_pressed('q'):
                break
            
            for msg in unpacker:
                print(msg[0])
                if msg[0].find("mot_move_homed") >= 0:
                    is_homed = True

            time.sleep(0.1)

        self.portX = port
        self.unpackerX = unpacker


    def initializeMotors(self, com_port_x, com_port_y):
        """
            Ports are initialized and homing is performed for both motors synchronously
        """
        portX = serial.Serial(com_port_x, 115200, rtscts=True, timeout=0.01)
        portX.rts = True
        portX.reset_input_buffer()
        portX.reset_output_buffer()
        portX.rts = False
        portX.write(apt.mot_move_home(source=self.HOST, dest=self.BAY0 ,chan_ident=self.CHANNEL))
        unpackerX = apt.Unpacker(portX)

        portY = serial.Serial(com_port_y, 115200, rtscts=True, timeout=0.01)
        portY.rts = True
        portY.reset_input_buffer()
        portY.reset_output_buffer()
        portY.rts = False
        portY.write(apt.mot_move_home(source=self.HOST, dest=self.BAY0 ,chan_ident=self.CHANNEL))
        unpackerY = apt.Unpacker(portY)

        self.portX = portX
        self.unpackerX = unpackerX
        self.portY = portY
        self.unpackerY = unpackerY

        is_homed_x = False
        is_homed_y = False
        while(not (is_homed_x and is_homed_y)):    
            if keyboard.is_pressed('q'):
                break
            
            for msg in self.unpackerX:
                print(msg[0])
                if msg[0].find("mot_move_homed") >= 0:
                    is_homed_x = True

            for msg in self.unpackerY:
                print(msg[0])
                if msg[0].find("mot_move_homed") >= 0:
                    is_homed_y = True
            time.sleep(0.1)


    def close_motors(self):
        self.portX.close()
        self.portY.close()
    
    def initializePM400(self):
        device_list = usbtmc.list_devices()
        print(device_list)

        self.instrument = usbtmc.Instrument(device_list[0])
        # set pm400 to power measurement mode
        self.instrument.write("MEASure:POWer")
        print(self.instrument)

    def closePM400(self):
        self.instrument.close()
        # port.write(apt.mot_move_absolute(source=1, dest=0x21, chan_ident=1, position=50000))
        


    def moveMotorAbsolute(self, distance_mm, motor_id):
        """
        distance_mm: absolute distance in mm
        motor_id: motor axis ("x" or "y")
        """
        if motor_id == "x":
            port = self.portX
            unpacker = self.unpackerX
        elif motor_id == "y":
            port = self.portY
            unpacker = self.unpackerY
        else:
            print("Wrong motor_id")
            return
        

        is_move_completed = False

        port.write(apt.mot_move_absolute(source=self.HOST, dest=self.BAY0, chan_ident=self.CHANNEL, position=int(distance_mm*self.MM_TO_ENCODER)))

        while(not is_move_completed):    
            if keyboard.is_pressed('q'):
                break
        
            for msg in unpacker: #in order to get the move completed message, homing should be performed before
                print(msg[0])
                if msg[0].find("mot_move_completed") >= 0:
                    is_move_completed = True
            time.sleep(0.1)

    def moveMotorRelativeAndMeasure(self, distance_mm, motor_id):
        """
        distance_mm: relative distance in mm
        motor_id: motor axis ("x" or "y")
        """
        if motor_id == "x":
            port = self.portX
            unpacker = self.unpackerX
        elif motor_id == "y":
            port = self.portY
            unpacker = self.unpackerY
        else:
            print("Wrong motor_id")
            return


        

        port.write(apt.mot_move_relative(source=self.HOST, dest=self.BAY0, chan_ident=self.CHANNEL, distance=int(distance_mm*self.MM_TO_ENCODER) ))


        is_move_completed = False
        measurements = []
        times = []
        start = time.time()

        while(not is_move_completed):    
            if keyboard.is_pressed('q'):
                break    
            
            self.instrument.write("READ?")
            reading = self.instrument.read()
            times.append(time.time() - start)
            measurements.append(reading)


            for msg in unpacker: #in order to get the move completed message, homing should be performed before
                print(msg[0])
                if msg[0].find("mot_move_completed") >= 0:
                    is_move_completed = True
        
        #self.instrument.clear()


        times = np.array(times, dtype='float64')
        measurements = np.array(measurements, dtype='float64')
        measurements_mW = measurements * 1000

        total_time = times[-1]
        speed = distance_mm / total_time # mm/s

        distances_mm = times * speed

        

        measurement_dict = {            
            "t_s": times,
            "measurements_mw": measurements_mW,
            "distances_mm": distances_mm,
            "speed_mms": speed}
        
        return measurement_dict


    def moveMotorsAbsolute(self, distance_mm_x, distance_mm_y):
        """
            Motors are moved to absolute positions distance_mm_x and distance_mm_y
        """
        is_move_completed_x = False
        is_move_completed_y = False

        self.portX.write(apt.mot_move_absolute(source=self.HOST, dest=self.BAY0, chan_ident=self.CHANNEL, position=int(distance_mm_x*self.MM_TO_ENCODER)))
        self.portY.write(apt.mot_move_absolute(source=self.HOST, dest=self.BAY0, chan_ident=self.CHANNEL, position=int(distance_mm_y*self.MM_TO_ENCODER)))

        while(not (is_move_completed_x and is_move_completed_y)):    
            if keyboard.is_pressed('q'):
                break
        
            for msg in self.unpackerX: #in order to get the move completed message, homing should be performed before
                print(msg[0])
                if msg[0].find("mot_move_completed") >= 0:
                    is_move_completed_x = True
            for msg in self.unpackerY: #in order to get the move completed message, homing should be performed before
                print(msg[0])
                if msg[0].find("mot_move_completed") >= 0:
                    is_move_completed_y = True
            time.sleep(0.1)



# controller = MotorAndPowerMeterController()

# (portX, unpackerX, portY, unpackerY) = controller.initializeMotors(COM_PORT_X, COM_PORT_Y)

# input("Press enter to start the move...")  


# controller.moveMotors(portX, unpackerX, 2, portY, unpackerY, 5)

# print("first move completed")
# controller.moveMotors(portX, unpackerX, 1, portY, unpackerY, 3)

# portX.close()
# portY.close()
