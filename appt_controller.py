# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 12:21:39 2023

@author: ki52soho
"""

from instrumental import instrument, list_instruments
import time
import keyboard

paramsets = list_instruments()

print(paramsets)

controller = instrument(paramsets[0])



controller.home()
print("homed")

controller.move_to("0 mm")

while(not abs(controller.get_position()) < 1e-3 ):
    print(controller.get_position())
    time.sleep(0.1)
    

input("Wait for enter...")

controller.move_to("2 mm")


while(True):
    if keyboard.is_pressed("q"):
        break
    
    print(controller.get_position())
    time.sleep(0.1)
    

controller.close()
print("closed")