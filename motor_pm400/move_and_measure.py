"""
Created on Thu Apr 20 16:06:10 2023

@author: ki52soho
"""

import thorlabs_apt_protocol as apt
import time
import serial
import keyboard
import usbtmc
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np


distance_mm = 30
distance_to_plane = "410mm"
distance_to_mirror_center = "d0"


COM_PORT = "COM5"
MM_TO_ENCODER = 25000 # from Z606 motorized actuator documentation

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




# initilaze motor controller
port = serial.Serial(COM_PORT, 115200, rtscts=True, timeout=0.1)
port.rts = True
port.reset_input_buffer()
port.reset_output_buffer()
port.rts = False
port.write(apt.mot_move_home(source=HOST, dest=BAY0 ,chan_ident=CHANNEL))

#initialize pm400
device_list = usbtmc.list_devices()
print(device_list)

instrument = usbtmc.Instrument(device_list[0])

print(instrument)

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
   
position_input_mm = input("Enter the position in mm\n")  




# set pm400 to power measurement mode
instrument.write("MEASure:POWer")

port.write(apt.mot_move_absolute(source=HOST, dest=BAY0, chan_ident=CHANNEL, position=int(distance_mm*MM_TO_ENCODER) ))



is_move_completed = False
measurements = []
times = []
start = time.time()

# packer_check_count = 0 #packer checking operation slows down the measurement loop
i = 0
while(is_move_completed is False):    
    
    
    instrument.write("READ?")
    reading = instrument.read()
    times.append(time.time() - start)
    measurements.append(reading)
    
    
    for msg in unpacker:      
        print(msg[0])
        if msg[0].find("mot_move_completed") >= 0:
            is_move_completed = True        
   
    
    
    
def interpolate_and_lowpass(t, x, T, N):
    """
    t: irregular time vector
    x: irregular measurement vector
    T: interpolation period
    N: moving average filter length
    """
    fcubic = interpolate.interp1d(t, x, kind='cubic')
    tnew = np.arange(0.005, t[-1], T)
    ycubic = fcubic(tnew)

    smooth_x = np.convolve(ycubic, np.ones(N)/N, mode='same')
    return tnew, smooth_x



#close motor controller
port.close()

#close pm400
instrument.clear()
instrument.close()



times = np.array(times, dtype='float64')
measurements = np.array(measurements, dtype='float64')
measurements_mW = measurements * 1000

total_time = times[-1]
speed = distance_mm / total_time # mm/s

distances_mm = times * speed




measurement_dict = {
    "position_mm": position_input_mm,
    "t_s": times,
    "measurements_mw": measurements_mW,
    "distances_mm": distances_mm,
    "speed_mms": speed
    }

with open('measurements_2/{}/{}/{}.pkl'.format(distance_to_mirror_center, distance_to_plane, position_input_mm), 'wb') as f:
    pickle.dump(measurement_dict, f)


# ##  Plotting

# interpolation_period = 0.01

# plt.subplot(2, 2, 1)
# plt.plot(times, measurements_mW)
# plt.ylabel("Power (mW)")
# plt.xlabel("t (s)")
# plt.title("Power-Time (Raw Sensor Data)")


# plt.subplot(2,2,2)
# flinear = interpolate.interp1d(times, measurements_mW)
# fcubic = interpolate.interp1d(times, measurements_mW, kind='cubic')

# xnew = np.arange(0.005, total_time, 0.01)
# ylinear = flinear(xnew)
# ycubic = fcubic(xnew)
# plt.plot(times, measurements_mW, 'X', xnew, ylinear, 'x', xnew, ycubic, 'o')
# plt.title("Power-Time (Cubic Interpolation Period:{})".format(interpolation_period))


# plt.subplot(2,2,3)
# N=10
# smooth_measurements_mW = np.convolve(ycubic, np.ones(N)/N, mode='same')
# plt.plot(xnew, smooth_measurements_mW)
# plt.ylabel("Power (mW)")
# plt.xlabel("time (s)")
# plt.title("Power-Time (Cubic Interpolation Smoothend Filter Length:{})".format(N))



# plt.subplot(2, 2, 4)
# plt.plot(distances_mm, measurements_mW)
# plt.ylabel("Power (mW)")
# plt.xlabel("distance (mm)")
# plt.title("Power-Distance (Raw Sensor Data)")


# plt.show()
