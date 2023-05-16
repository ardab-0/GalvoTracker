# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 12:10:45 2023

@author: ki52soho
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:40:39 2023

@author: ki52soho
"""


import optoMDC
import time
import keyboard
import numpy as np
from coordinate_transformation import CoordinateTransform




MAX_XY = 10 # mm
SLEEP_DURATION = 0.5 # s



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



# Display a pattern

y_t = np.zeros(11)
x_t = np.linspace(0, 100, 11) 
# x_t = np.array([0])
# y_t = np.array([0])


# x_t = np.zeros(11)
# y_t = np.linspace(0, 100, 11) 



coordinate_transform = CoordinateTransform(d=0, D=200, rotation_degree=45)

x_m, y_m = coordinate_transform.target_to_mirror(x_t, y_t)

print("x_t", x_t)
print("y_t", y_t)

print("x_m", x_m)
print("y_m", y_m)
i=0

while True:
    try:
        if keyboard.is_pressed('q'):
            break
        si_0.SetXY(x_m[i])        
        si_1.SetXY(y_m[i])                      
        #time.sleep(SLEEP_DURATION)
        
        i = (i + 1) % len(x_m)
        input("Press Enter To Continue")    
        
    except:
        break



si_0.SetXY(0)        
si_1.SetXY(0)
mre2.disconnect()
print("done")


