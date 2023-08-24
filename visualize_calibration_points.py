from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import optimal_rotation_and_translation

CALIBRATION_ITER = 10

def main():
    save_path = "ir_calibration_parameters"


    fig = plt.figure()
    ax = plt.axes(projection='3d')

    with open('{}/parameters.pkl'.format(save_path), 'rb') as f:
        loaded_dict = pickle.load(f)
        R = loaded_dict["R"]
        t = loaded_dict["t"]
        laser_points = loaded_dict["laser_points"]
        camera_points = loaded_dict["camera_points"]

    laser_points = laser_points[:, :3*CALIBRATION_ITER]
    camera_points = camera_points[:, :3*CALIBRATION_ITER]
    R, t = optimal_rotation_and_translation(camera_points, laser_points)

    print(R)
    print("\n\n")
    print(t)
    cam_wrt_laser = R @ camera_points + t


    ax.scatter3D(laser_points[0], laser_points[1], laser_points[2], color="red")

    ax.scatter3D(camera_points[0], camera_points[1], camera_points[2], color="green")

    ax.scatter3D(cam_wrt_laser[0], cam_wrt_laser[1], cam_wrt_laser[2], color="blue")
    
    print("Error: ", np.sqrt(np.mean(np.square(cam_wrt_laser-laser_points))))

    plt.legend(("laser points", "camera points", "camera points after transform"))
    plt.show()


if __name__ == "__main__":
    main()