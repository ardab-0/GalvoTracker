# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:06:10 2023

@author: ki52soho
"""

import thorlabs_apt_protocol as apt
import time
import serial
import keyboard


COM_PORT = "COM5"

ENCODER_TO_MM = 25000 # from Z606 motorized actuator documentation

# from endpoint enums in thorlabs_apt_device library
HOST = 0x01
RACK = 0x11
BAY0 = 0x21
BAY1 = 0x22
BAY2 = 0x23
BAY3 = 0x24
BAY4 = 0x25
BAY5 = 0x26
BAY6 = 0x27
BAY7 = 0x28
BAY8 = 0x29
BAY9 = 0x2A
USB = 0x50

CHANNEL = 1 # first channel in controller (there is one channel in tdc001)
# from endpoint enums in thorlabs_apt_device library





port = serial.Serial(COM_PORT, 115200, rtscts=True, timeout=0.1)
port.rts = True
port.reset_input_buffer()
port.reset_output_buffer()
port.rts = False
port.write(apt.mot_move_home(source=HOST, dest=BAY0 ,chan_ident=CHANNEL))





# port.write(apt.mot_move_absolute(source=1, dest=0x21, chan_ident=1, position=50000))
is_homed = False
while(not is_homed):    
    if keyboard.is_pressed('q'):
        break
    unpacker = apt.Unpacker(port)
    for msg in unpacker:
        print(msg[0])
        if msg[0].find("mot_move_homed") >= 0:
            is_homed = True
    time.sleep(0.1)
    
input("Press enter to start the move...")  

distance_mm = 5
is_move_completed = False
port.write(apt.mot_move_absolute(source=HOST, dest=BAY0, chan_ident=CHANNEL, position=distance_mm*ENCODER_TO_MM))
while(not is_move_completed):    
    if keyboard.is_pressed('q'):
        break
    
    for msg in unpacker: #in order to get the move completed message, homing should be performed before
        print(msg[0])
        if msg[0].find("mot_move_completed") >= 0:
            is_move_completed = True
    time.sleep(0.1)

port.close()



