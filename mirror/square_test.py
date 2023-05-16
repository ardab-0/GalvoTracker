# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:15:34 2023

@author: ki52soho
"""

import optoMDC
import time
import keyboard
import numpy as np
from coordinate_transformation import CoordinateTransform


MAX_XY = 10 # mm
SLEEP_DURATION = 2 # s



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

x_t = np.array([5*MAX_XY, -MAX_XY, -MAX_XY, 5*MAX_XY]) 
y_t = np.array([5*MAX_XY, 5*MAX_XY, -MAX_XY, -MAX_XY]) 



coordinate_transform = CoordinateTransform(d=1.3, D=250, rotation_degree=45)

x_m, y_m = coordinate_transform.target_to_mirror(x_t, y_t)
print("x_m", x_m)
print("y_m", y_m)

i=0
while True:
    try:
        if keyboard.is_pressed('q'):
            break
        si_0.SetXY(x_m[i])        
        si_1.SetXY(y_m[i])                      
        time.sleep(SLEEP_DURATION)
        
        i = (i + 1) % len(x_m)        
    except:
        break



si_0.SetXY(0)        
si_1.SetXY(0)
mre2.disconnect()
print("done")