import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

MAX_CALIBRATION_ITER = 10
MIN_CALIBRATION_ITER = 2
FIGURE_SAVE_FOLDER = "calibration_result_figures/"

ACCURACY_PATH = "calibration_accuracy_727mm"
save_path = "ir_calibration_parameters"

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

    
    
    fig = plt.figure(figsize=(16,8))

    ax = fig.add_subplot(1,2,1)
    ax.scatter(camera_points[0, :3*i:3], camera_points[2, :3*i:3])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel("z (mm)")
    ax.set_title("Calibration Plate Positions")
    
    ax = fig.add_subplot(1,2,2, projection='3d')
    surf = ax.plot_surface(X, Y, distance_mm, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("RMSE (mm)")
    ax.set_title(f"RMSE with {i} calibration positions. Average Error Distance: {avg_distance_mm:.2f}mm")
    
    distance_error_mm.append(avg_distance_mm)
    calibration_iter.append(i)
    plt.savefig(f"{FIGURE_SAVE_FOLDER}{ACCURACY_PATH}_iter{i}.png")
    plt.show()




plt.plot(calibration_iter, distance_error_mm)
plt.xlabel("Calibration iteration")
plt.ylabel("Average Error Distance (mm)")
plt.grid()
plt.savefig(f"{FIGURE_SAVE_FOLDER}{ACCURACY_PATH}_rmse-iter.png")
plt.show()




