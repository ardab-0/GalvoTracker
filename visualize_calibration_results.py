import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

MAX_CALIBRATION_ITER = 10
MIN_CALIBRATION_ITER = 2
FIGURE_SAVE_FOLDER = "Figures/calibration_result_figures/"

ACCURACY_PATH = "calibration_accuracy_results/calibration_accuracy_727mm"
save_path = "calibration_parameters"

distance_error_mm = []
calibration_iter = []

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    data[s>=m] = mdev
    return data


for i in range(MIN_CALIBRATION_ITER, MAX_CALIBRATION_ITER+1):
    with open('{}/iter_{}.pkl'.format(ACCURACY_PATH, i), 'rb') as f:
        accuracy_results = pickle.load(f)


    with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
        loaded_dict = pickle.load(f)
        laser_points = loaded_dict["laser_points"]
        camera_points = loaded_dict["camera_points"]

    # accuracy_results = {"x_t": x_t,
    #                     "y_t": y_t,
    #                     "z_t": z_t,
    #                     "rmse": rmse_scores}

    # x_t = accuracy_results["pointer_pos"][:, 0]
    # y_t = accuracy_results["pointer_pos"][:, 1]

    x_t = accuracy_results["x_t"]
    y_t = accuracy_results["y_t"]
    z_t = accuracy_results["z_t"]
    rmse = accuracy_results["rmse"]
    

    w = int(np.sqrt(len(rmse)))

    X, Y =np.array(x_t).reshape((w, w)), np.array(y_t).reshape((w, w))
    rmse = np.array(rmse).reshape((w, w))
    distance_mm = rmse * np.sqrt(3)
    
    distance_mm = reject_outliers(distance_mm)

    avg_distance_mm = np.mean(distance_mm)

    
    
    

    
    plt.scatter(camera_points[0, :3*i:3], camera_points[2, :3*i:3])
    plt.xlabel('x (mm)')
    plt.ylabel("z (mm)")
    plt.title("Calibration Plate Positions")
    
    
    
    plt.savefig(f"{FIGURE_SAVE_FOLDER}_iter{i}.png")
    plt.show()




plt.plot(calibration_iter, distance_error_mm)
plt.xlabel("Calibration iteration")
plt.ylabel("Average Error Distance (mm)")
plt.grid()
plt.savefig(f"{FIGURE_SAVE_FOLDER}{ACCURACY_PATH}_rmse-iter.png")
plt.show()




