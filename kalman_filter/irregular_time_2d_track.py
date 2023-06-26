from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import predict, update
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse


class Sensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), acc=(0.0, 0.0), acc_noise=(0.5, 0.5), dt_mean=1, dt_std=0.1):
        self.vel = np.array(vel, dtype=float)
        self.acc_std = np.array(acc_noise, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.acc = np.array(acc, dtype=float)
        self.acc_update_interval = dt_mean
        self.time_since_last_update = dt_mean
        self.acc_diff = 0
        self.dt_mean = dt_mean
        self.dt_std = dt_std
        self.dt = 0


    def update(self):
        self.dt = self.dt_mean + np.random.randn() * self.dt_std
        self.acc_diff = np.random.randn(*self.acc.shape) * self.acc_std
        # if self.time_since_last_update >= self.acc_update_interval:
        #     self.acc_diff = np.random.randn(*self.acc.shape) * self.acc_std
        #     self.time_since_last_update = 0
        # else:
        #     self.time_since_last_update += self.dt

        self.acc += self.acc_diff * self.dt
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt


        return self.dt

    def read_pos(self):
        return [self.pos[0], self.pos[1]]

    def read_vel(self):
        return [self.vel[0], self.vel[1]]

    def read_acc(self):
        return [self.acc[0], self.acc[1]]


def predict_position(x, dt):
    A = np.array(
        [
            [1, dt, 0.5 * dt * dt, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt * dt],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    x_pred = A @ x
    return x_pred


def print_history(state_hist, pos_hist, vel_hist, acc_hist, t):

    time = 0
    for i in range(len(state_hist)):
        time += t[i]
        print(f"Time: {time} s")
        print(
            f"State Pos X: {state_hist[i][0,0]}, Sensor Pos X: {pos_hist[i][0]}, State Pos Y: {state_hist[i][3,0]}, Sensor Pos Y: {pos_hist[i][1]}"
        )
        print(
            f"State Vel X: {state_hist[i][1,0]}, Sensor Vel X: {vel_hist[i][0]}, State Vel Y: {state_hist[i][4,0]}, Sensor Vel Y: {vel_hist[i][1]}"
        )
        print(
            f"State Acc X: {state_hist[i][2,0]}, Sensor Acc X: {acc_hist[i][0]}, State Acc Y: {state_hist[i][5,0]}, Sensor Acc Y: {acc_hist[i][1]}"
        )
        print("")




class SecondOrderKF():
    """Create second order Kalman filter.
    Specify R and Q as floats."""
    def __init__(self, R_std, Q_std, P_std=10):
        self.R_std = R_std
        self.Q_std = Q_std
        self.P_std = P_std        
        self.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
        self.R = np.eye(2) * R_std**2        
        self.x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.P = np.eye(6) * P_std**2


    def update(self, dt, z):
        self.F = np.array(
            [
                [1, dt, 0.5 * dt * dt, 0, 0, 0],
                [0, 1, dt, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, dt, 0.5 * dt * dt],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
        self.Q = block_diag(q, q)
        self.x, self.P = predict(self.x, self.P, self.F, self.Q)
        self.x, self.P = update(self.x, self.P, z, self.R, self.H)

        return self.x, self.P



R_std = 0.5
Q_std = 2
dt_mean = 0.033
dt_std = 0.001
next_t = 3 * dt_mean

# simulate target movement
N = 300
sensor = Sensor((0, 0), (20, 0.2), acc_noise=(100,100), dt_mean=dt_mean, dt_std=dt_std)
tracker = SecondOrderKF(R_std=R_std, Q_std=Q_std, P_std=10)

zs = []
vels = []
accs = []
mu = []
dts = []

for i in range(N):
    dt = sensor.update()
    dts.append(dt)
    zs.append(sensor.read_pos())
    vels.append(sensor.read_vel())
    accs.append(sensor.read_acc())


    x, P = tracker.update(dt, sensor.read_pos())
    cov = np.array([[P[0, 0], P[3, 0]], [P[0, 3], P[3, 3]]])
    mean = (x[0, 0], x[3, 0])
    x_pred = predict_position(x, next_t)
    plt.plot(x_pred[0, 0], x_pred[3, 0], marker="v", color="red")
    plot_covariance_ellipse(mean, cov=cov, fc="g", std=3, alpha=0.5)
    mu.append(x)

zs = np.array(zs)
vels = np.array(vels)
accs = np.array(accs)
mu = np.array(mu)

print_history(mu, pos_hist=zs, vel_hist=vels, acc_hist=accs, t=dts)

# plot results
plt.plot(mu[:, 0], mu[:, 3])
plt.plot(zs[:, 0], zs[:, 1])
plt.legend(loc=2)
plt.show()
