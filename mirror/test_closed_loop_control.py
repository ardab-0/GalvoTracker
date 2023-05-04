# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:30:29 2023

@author: ki52soho
"""
import optoMDC
import time
import keyboard


MAX_XY = 0.1


mre2 = optoMDC.connect()
mre2.reset()


ch_0 = mre2.Mirror.Channel_0

ch_0.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
#ch_0.InputConditioning.SetGain(1.0)                  # (2) here we tell the Manager some input conditioning parameters
ch_0.SetControlMode(optoMDC.Units.XY)           
#ch_0.LinearOutput.SetCurrentLimit(0.7)               # (4) here we tell the Manager to limit the current to 700mA (default)

ch_0.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.


si_0 = mre2.Mirror.Channel_0.StaticInput


ch_1 = mre2.Mirror.Channel_1

ch_1.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
#ch_0.InputConditioning.SetGain(1.0)                  # (2) here we tell the Manager some input conditioning parameters
ch_1.SetControlMode(optoMDC.Units.XY)           
#ch_0.LinearOutput.SetCurrentLimit(0.7)               # (4) here we tell the Manager to limit the current to 700mA (default)

ch_1.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.


si_1 = mre2.Mirror.Channel_1.StaticInput






while True:
    try:
        if keyboard.is_pressed('q'):
            break
        si_0.SetXY(MAX_XY)        
        si_1.SetXY(MAX_XY)                      
        time.sleep(1)
        
        
        si_0.SetXY(-MAX_XY)        
        si_1.SetXY(MAX_XY)                      
        time.sleep(1)
        
        
        si_0.SetXY(-MAX_XY)        
        si_1.SetXY(-MAX_XY)                      
        time.sleep(1)
        
        
        si_0.SetXY(MAX_XY)        
        si_1.SetXY(-MAX_XY)                      
        time.sleep(1)
        
        
    except:
        break




mre2.disconnect()
print("done")


