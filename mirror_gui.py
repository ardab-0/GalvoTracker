import tkinter as tk
import optoMDC
from mirror.coordinate_transformation import CoordinateTransform
import numpy as np
import time

def show_values(event):
    print (w1.get(), w2.get())


def update_mirror(event):
    start = time.time()
    # time.sleep(0.030)
    y_t = np.array([w2.get()])
    x_t = np.array([w1.get()])
    coordinate_transform = CoordinateTransform(d=0, D=w3.get(), rotation_degree=45)
    y_m, x_m = coordinate_transform.target_to_mirror(y_t, x_t) # order is changed in order to change x and y axis
    si_0.SetXY(y_m[0])        
    si_1.SetXY(x_m[0]) 

    print("frame time ", time.time() - start)



# initialize mirrors
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




 

master = tk.Tk()
w1 = tk.Scale(master, from_=-50, to=50, tickinterval=1, command=update_mirror)
w1.set(0)
w1.pack()
w2 = tk.Scale(master, from_=-50, to=50,tickinterval=1, command=update_mirror)
w2.set(0)
w2.pack()
w3 = tk.Scale(master, from_=20, to=1000,tickinterval=1,  command=update_mirror)
w3.set(300)
w3.pack()

tk.Button(master, text='Show', command=show_values).pack()

master.mainloop()


mre2.disconnect()
print("done")