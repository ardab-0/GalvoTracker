
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:04:41 2023

@author: ki52soho
"""

import optoMDC
import keyboard



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





while(True):
    try:
        if keyboard.is_pressed('q'):
            break
        si_0.SetXY(0)        
        si_1.SetXY(0)                    
        #time.sleep(SLEEP_DURATION)
        
       
           
        
    except:
        break
    
    
mre2.disconnect()
print("done")