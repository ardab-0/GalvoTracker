import os
import numpy as np
import pickle

D = 400
d = 0
measurement_foldername = "measurements_3"


def create_folder_structure(path): 
    if not os.path.exists(path):   
        os.makedirs(path)


# create necessary folder structur if necessary
distance_to_plane = "{}mm".format(D) 
distance_to_mirror_center = "d{}".format(d)
save_path = "{}/{}/{}/".format(measurement_foldername, distance_to_plane, distance_to_mirror_center)
create_folder_structure(save_path)




x_t = np.array([-3.5, -2, -1, 0, 1, 2, 3])
y_t = np.array([-3, -2, -1, 0, 1, 2, 3])

# Set mirror position
for i in range(len(x_t)):
    for j in range(len(y_t)):

        
        measurement_dictionary = {}

        position_input_mm = "{}x{}".format(x_t[i], y_t[j])

        with open('{}{}.pkl'.format(save_path, position_input_mm), 'wb') as f:
            pickle.dump(measurement_dictionary, f)



for i in range(len(x_t)):
    for j in range(len(y_t)):

        

        position_input_mm = "{}x{}".format(x_t[i], y_t[j])

        with open('{}{}.pkl'.format(save_path, position_input_mm), 'rb') as f:
            print(i, j)
            loaded_dict = pickle.load(f)
            print(loaded_dict)