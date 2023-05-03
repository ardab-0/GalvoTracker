# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:37:25 2023

@author: ki52soho
"""

# The BBD201 DC motor controller has a dedicated class which handles its specifics
from thorlabs_apt_device import TDC001
from thorlabs_apt_device.utils import from_pos, to_pos
import time
import keyboard
from functools import partial





# Register for callbacks in case the device reports an error
def error_callback(source, msgid, code, notes):
    # Hopefully never see this!
    print(f"Device {source} reported error code {code}: {notes}")


PORT = "COM5"






# You can try to find a device automatically:
stage = TDC001(PORT)
# Or, if you know the serial number of the device starts with "73123":
# stage = BBD201(serial_number="73123")
# You can also specify the serial port device explicitly.
# On Windows, your serial port may be called COM3, COM5 etc.
# stage = BBD201("/dev/ttyUSB0")

# Flash the LED on the device to identify it
stage.identify()

stage.register_error_callback(error_callback)


# Build our custom coversions using mm
from_mm = partial(from_pos, factor=25000)
to_mm = partial(from_pos, factor=1/25000)


print(stage.status)



# Perform moves in mm instead of encoder counts
stage.move_absolute(from_mm(2))



# Check position in mm

while(True):
    if keyboard.is_pressed("q"):
        break
    
    print(to_mm(stage.status["position"]))
    time.sleep(0.1)
#See the position (in encoder counts)

print(stage.status)

stage.close()
print("closed")
