# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:27:57 2023

@author: ki52soho
"""


import optoMDC
import time
import keyboard
import numpy as np
from coordinate_transformation import CoordinateTransform


# Constants

MAX_XY = 20
P = 50
DELTA = 0.01


# Constants

mre2 = optoMDC.connect()
mre2.reset()

# Set up mirror in closed loop control mode(XY)
ch_0 = mre2.Mirror.Channel_0
ch_0.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
ch_0.SetControlMode(optoMDC.Units.XY)           
ch_0.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_0 = mre2.Mirror.Channel_0.StaticInput


ch_1 = mre2.Mirror.Channel_1
ch_1.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
ch_1.SetControlMode(optoMDC.Units.XY)           
ch_1.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.
si_1 = mre2.Mirror.Channel_1.StaticInput



# Display a pattern
theta = np.linspace(0, 2*np.pi, 2*P)

x_t = np.cos(theta) 
y_t = np.sin(theta) 





coordinate_transform = CoordinateTransform(d=1.3, D=140, rotation_degree=45)
x_m, y_m = coordinate_transform.target_to_mirror(x_t, y_t)


i = 0

while True:
    try:
        if keyboard.is_pressed('q'):
            break
        
        
        si_0.SetXY(MAX_XY * x_m[i])        
        si_1.SetXY(MAX_XY * y_m[i])                      
        time.sleep(DELTA)
        
        i = (i + 1) % (2*P)
        
        
        
        
    except:
        break




mre2.disconnect()
print("done")


