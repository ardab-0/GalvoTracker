import sys
import machine

led = machine.Pin("LED", machine.Pin.OUT)
sensor_temp = machine.ADC(4)
conversion_factor = 3.3 / (65535)

sensor_light_1 = machine.ADC(0)
sensor_light_2 = machine.ADC(1)
sensor_light_3 = machine.ADC(2)

def read_photodiode_1():
    reading_voltage = sensor_light_1.read_u16() * conversion_factor
    print(reading_voltage)
    
def read_photodiode_2():
    reading_voltage = sensor_light_2.read_u16() * conversion_factor
    print(reading_voltage)
    
def read_photodiode_3():
    reading_voltage = sensor_light_3.read_u16() * conversion_factor
    print(reading_voltage)



while True:
    # read a command from the host
    v = sys.stdin.readline().strip()
    
    # perform the requested action
    
    if v.lower() == "pd_1":
        read_photodiode_1()
    elif v.lower() == "pd_2":
        read_photodiode_2()
    elif v.lower() == "pd_3":
        read_photodiode_3()  


