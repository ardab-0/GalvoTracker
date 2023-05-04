import usbtmc
import matplotlib.pyplot as plt
import time
import numpy as np

device_list = usbtmc.list_devices()
print(device_list)

instrument = usbtmc.Instrument(device_list[0])

print(instrument)

# instrument.write("*IDN?")
# print(instrument.read())
instrument.write("MEASure:POWer")
start = time.time()
measurements = []
times = []
for i in range(1000):
    # instrument.write("MEASure:POWer")
    instrument.write("READ?")
    reading = instrument.read()
    print(reading)
    measurements.append(reading)
    times.append(time.time() - start)

end = time.time()
print(end - start)


instrument.clear()
instrument.close()

times = np.array(times, dtype='float64')
measurements = np.array(measurements, dtype='float64')
measurements_mW = measurements * 1000

print(times)
plt.plot(times, measurements_mW)
plt.show()