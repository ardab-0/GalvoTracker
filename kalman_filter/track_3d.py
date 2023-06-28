from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import predict, update
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse


class Sensor(object):
    def __init__(self, pos=(0, 0, 0), vel=(0, 0, 0), acc=(0.0, 0.0, 0.0), acc_noise=(0.5, 0.5, 0.5), dt_mean=1.0, dt_std=0.1):
        self.vel = np.array(vel, dtype=float)
        self.acc_std = np.array(acc_noise, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.acc = np.array(acc, dtype=float)
        self.acc_update_interval = 5*dt_mean
        self.time_since_last_update = 5*dt_mean
        self.acc_diff = 0
        self.dt_mean = dt_mean
        self.dt_std = dt_std
        self.dt = 0


    def update(self):
        self.dt = self.dt_mean + np.random.randn() * self.dt_std
        if self.time_since_last_update >= self.acc_update_interval:
            self.acc_diff = np.random.randn(*self.acc.shape) * self.acc_std
            self.time_since_last_update = 0
        else:
            self.time_since_last_update += self.dt

        self.acc = self.acc_diff * self.dt
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt


        return self.dt

    def read_pos(self):
        return [self.pos[0], self.pos[1], self.pos[2]]

    def read_vel(self):
        return [self.vel[0], self.vel[1],  self.vel[2]]

    def read_acc(self):
        return [self.acc[0], self.acc[1], self.acc[2]]


def predict_position(x, dt):
    A = np.array(
            [
                [1, dt, 0.5 * dt * dt, 0, 0, 0, 0, 0, 0],
                [0, 1, dt, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt * dt, 0, 0, 0],
                [0, 0, 0, 0, 1, dt, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt * dt],
                [0, 0, 0, 0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]
        )

    x_pred = A @ x
    return x_pred


def print_history(state_hist, pos_hist, vel_hist, acc_hist, t):
    if len(state_hist[0]) == 9:
        time = 0
        for i in range(len(state_hist)):
            time += t[i]
            print(f"Time: {time} s")
            print(
                f"State Pos X: {state_hist[i][0,0]}, Sensor Pos X: {pos_hist[i][0]}, State Pos Y: {state_hist[i][3,0]}, Sensor Pos Y: {pos_hist[i][1]}, State Pos Z: {state_hist[i][6,0]}, Sensor Pos Z: {pos_hist[i][2]}"
            )
            print(
                f"State Vel X: {state_hist[i][1,0]}, Sensor Vel X: {vel_hist[i][0]}, State Vel Y: {state_hist[i][4,0]}, Sensor Vel Y: {vel_hist[i][1]}, State Vel Z: {state_hist[i][7,0]}, Sensor Vel Z: {vel_hist[i][2]}"
            )
            print(
                f"State Acc X: {state_hist[i][2,0]}, Sensor Acc X: {acc_hist[i][0]}, State Acc Y: {state_hist[i][5,0]}, Sensor Acc Y: {acc_hist[i][1]}, State Acc Z: {state_hist[i][8,0]}, Sensor Acc Z: {acc_hist[i][2]}"
            )
            print("")
    elif len(state_hist[0]) == 6:
        time = 0
        for i in range(len(state_hist)):
            time += t[i]
            print(f"Time: {time} s")
            print(
                f"State Pos X: {state_hist[i][0,0]}, Sensor Pos X: {pos_hist[i][0]}, State Pos Y: {state_hist[i][2,0]}, Sensor Pos Y: {pos_hist[i][1]}, State Pos Z: {state_hist[i][4,0]}, Sensor Pos Z: {pos_hist[i][2]}"
            )
            print(
                f"State Vel X: {state_hist[i][1,0]}, Sensor Vel X: {vel_hist[i][0]}, State Vel Y: {state_hist[i][3,0]}, Sensor Vel Y: {vel_hist[i][1]}, State Vel Z: {state_hist[i][5,0]}, Sensor Vel Z: {vel_hist[i][2]}"
            )
            print(
            )
            print("")

class FirstOrderKF():
    """Create second order Kalman filter.
    Specify R and Q as floats."""
    def __init__(self, R_std, Q_std, P_std=10):
        self.R_std = R_std
        self.Q_std = Q_std
        self.P_std = P_std        
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])
        self.R = np.eye(3) * R_std**2        
        self.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.P = np.eye(6) * P_std**2

    def predict_position(self, x, dt):
        A = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]
            ])
        x_pred = A @ x
        return x_pred

    def update(self, dt, z):
        self.F = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1]
            ]
        )
        q = Q_discrete_white_noise(dim=2, dt=dt, var=self.Q_std**2)
        self.Q = block_diag(q, q, q)
        self.x, self.P = predict(self.x, self.P, self.F, self.Q)
        self.x, self.P = update(self.x, self.P, z, self.R, self.H)

        return self.x, self.P

class SecondOrderKF():
    """Create second order Kalman filter.
    Specify R and Q as floats."""
    def __init__(self, R_std, Q_std, P_std=10):
        self.R_std = R_std
        self.Q_std = Q_std
        self.P_std = P_std        
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]])
        self.R = np.eye(3) * R_std**2        
        self.x = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
        self.P = np.eye(9) * P_std**2

    def predict_position(self, x, dt):
        A =np.array(
            [
                [1, dt, 0.5 * dt * dt, 0, 0, 0, 0, 0, 0],
                [0, 1, dt, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt * dt, 0, 0, 0],
                [0, 0, 0, 0, 1, dt, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt * dt],
                [0, 0, 0, 0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]
        )
        x_pred = A @ x
        return x_pred

    def update(self, dt, z):
        self.F = np.array(
            [
                [1, dt, 0.5 * dt * dt, 0, 0, 0, 0, 0, 0],
                [0, 1, dt, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt * dt, 0, 0, 0],
                [0, 0, 0, 0, 1, dt, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, dt, 0.5 * dt * dt],
                [0, 0, 0, 0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]
            ]
        )
        q = Q_discrete_white_noise(dim=3, dt=dt, var=self.Q_std**2)
        self.Q = block_diag(q, q, q)
        self.x, self.P = predict(self.x, self.P, self.F, self.Q)
        self.x, self.P = update(self.x, self.P, z, self.R, self.H)

        return self.x, self.P


def main():
    R_std = 0.1
    Q_std = 50
    dt_mean = 0.033 # 33ms
    dt_std = 0.001 # 1ms
    next_steps = 1
    next_t = next_steps * dt_mean
    filter_order = 2  # 1 or 2


    np.random.seed(100)
    # simulate target movement
    N = 300
    sensor = Sensor((0, 0, 0), (0, 0, 0), acc_noise=(10000, 10000, 10000), dt_mean=dt_mean, dt_std=dt_std)

    if filter_order == 1:
        tracker = FirstOrderKF(R_std=R_std, Q_std=Q_std, P_std=10)
    elif filter_order ==2:
        tracker = SecondOrderKF(R_std=R_std, Q_std=Q_std, P_std=10)

    zs = []
    vels = []
    accs = []
    mu = []
    dts = []
    preds = []

    for i in range(N):
        dt = sensor.update()
        dts.append(dt)
        zs.append(sensor.read_pos())
        vels.append(sensor.read_vel())
        accs.append(sensor.read_acc())


        x, P = tracker.update(dt, sensor.read_pos())
        # cov = np.array([[P[0, 0], P[2, 0]], [P[0, 2], P[2, 2]]])
        # mean = (x[0, 0], x[2, 0])
        x_pred = tracker.predict_position(x, next_t)
        preds.append(x_pred)
        #plot_covariance_ellipse(mean, cov=cov, fc="g", std=3, alpha=0.5)
        mu.append(x)

    zs = np.array(zs)
    vels = np.array(vels)
    accs = np.array(accs)
    mu = np.array(mu)
    preds = np.squeeze(np.array(preds))

    print_history(mu, pos_hist=zs, vel_hist=vels, acc_hist=accs, t=dts)

    if filter_order == 1:
        x_err = preds[:-next_steps, 0] - zs[next_steps:, 0]
        y_err = preds[:-next_steps, 2] - zs[next_steps:, 1]
        z_err = preds[:-next_steps, 4] - zs[next_steps:, 2]

        rms_error = np.sqrt(np.mean(np.square(x_err) + np.square(y_err) + np.square(z_err)))

        print("Error RMS: ", rms_error)
        # plot results
        plt.plot(mu[:, 0], mu[:, 2])
        plt.plot(zs[:, 0], zs[:, 1])
        plt.plot(preds[:, 0], preds[:, 2], marker="v", color="red")

    elif filter_order == 2:
        x_err = preds[:-next_steps, 0] - zs[next_steps:, 0]
        y_err = preds[:-next_steps, 3] - zs[next_steps:, 1]
        z_err = preds[:-next_steps, 6] - zs[next_steps:, 2]

        rms_error = np.sqrt( np.mean( np.square(x_err) + np.square(y_err) + np.square(z_err)) )

        print("Error RMS: ", rms_error )
        # plot results
        plt.plot(mu[:, 0], mu[:, 3])
        plt.plot(zs[:, 0], zs[:, 1])
        plt.plot(preds[:, 0], preds[:, 3], marker="v", color="red")

    plt.legend(("filter state", "measurement", "prediction"))
    plt.show()

if __name__ == "__main__":
    main()