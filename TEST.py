# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 12:02:46 2022

@author: andreasr
"""

#%% Check serial ports

import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))
    

#%% Board Connection
import optoMDC
from optoMDC.mre2 import MRE2Board
mre2 = MRE2Board(port='COM3', verbose=True)


mre2.Mirror.Channel_0.SetControlMode(2)
mre2.Mirror.Channel_0.GetControlMode()
mre2.Mirror.Channel_1.SetControlMode(2)
mre2.Mirror.Channel_1.GetControlMode()
sig_gen = mre2.Mirror.Channel_0.SignalGenerator
sig_gen1 = mre2.Mirror.Channel_1.SignalGenerator
sig_gen.SetUnit(optoMDC.Units.XY)
sig_gen1.SetUnit(optoMDC.Units.XY)
mre2.set_value(sig_gen.unit, 2)
mre2.set_value(sig_gen1.unit, 2)
mre2.get_value(sig_gen1.unit)
mre2.Mirror.Channel_0.StaticInput.SetAsInput()
mre2.Mirror.Channel_1.StaticInput.SetAsInput()
mre2.Mirror.Channel_0.StaticInput.SetXY(0.1)
mre2.Mirror.Channel_1.StaticInput.SetXY(0.0)
mre2.Mirror.Channel_0.StaticInput.GetXY()
mre2.Mirror.Channel_1.StaticInput.GetXY()
mre2.Mirror.Channel_1.SignalGenerator.Run()
mre2.Mirror.Channel_1.SignalGenerator.Stop()

mre2.Mirror.Channel_1.SetControlMode(2)

mre2.disconnect()



# #%% Mirror Connection

# from optoMDC.mre2 import MRE2Board
# mre2 = MRE2Board(port='COM3', verbose=True)

# import optoMDC


# mre2 = optoMDC.connect()
# mre2.Connection.disconnect()
# mre2.Connection.connect()
# mre2.Mirror.GetConnectedStatus()
# mre2.Mirror.Channel_0.GetControlMode()

# mre2.Mirror.Channel_0.StaticInput.GetXY()
# mre2.Mirror.Channel_0.StaticInput.SetXY(0)

#%% 

