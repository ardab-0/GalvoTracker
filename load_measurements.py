import pickle

distance_to_plane = "440mm"
distance_to_mirror_center = "d0"


def compare_measurements(n):
    for i in range(n):
        with open('measurements/{}/{}/{}.pkl'.format(distance_to_mirror_center, distance_to_plane, str(-1*i)), 'rb') as f:
            loaded_dict = pickle.load(f)
            print(str(-1*i) + ":   " + str(loaded_dict["maximum_pos_mm"]))


compare_measurements(6)



with open('measurements/{}/{}/{}.pkl'.format(distance_to_mirror_center, distance_to_plane, 0), 'rb') as f:
            loaded_dict = pickle.load(f)
            print(str(loaded_dict["t_s"]))