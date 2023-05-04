# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:08:06 2023

@author: ki52soho
"""
import time
import optoMDC
mre2 = optoMDC.connect()

ch_0 = mre2.Mirror.Channel_0

ch_0.StaticInput.SetAsInput()                        # (1) here we tell the Manager that we will use a static input
ch_0.InputConditioning.SetGain(1.0)                  # (2) here we tell the Manager some input conditioning parameters
ch_0.SetControlMode(optoMDC.Units.CURRENT)           # (3) here we tell the Manager that our input will be in units of current
ch_0.LinearOutput.SetCurrentLimit(0.7)               # (4) here we tell the Manager to limit the current to 700mA (default)

ch_0.Manager.CheckSignalFlow()                       # This is a useful method to make sure the signal flow is configured correctly.


si_0 = mre2.Mirror.Channel_0.StaticInput

si_0.SetCurrent(-0.075)                              # here we set a static output of 75mA. (Control mode above must also be CURRENT!)
time.sleep(1)
si_0.SetCurrent(0.0)                              # here we set a static output of 75mA. (Control mode above must also be CURRENT!)
time.sleep(1)
si_0.SetCurrent(0.075)                              # here we set a static output of 75mA. (Control mode above must also be CURRENT!)
time.sleep(1)
mre2.disconnect()
print("done")