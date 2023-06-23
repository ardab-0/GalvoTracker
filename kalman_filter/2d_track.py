from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse


class Sensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), acc=(0.0, 0.0), acc_noise=(0.5, 0.5), dt=1):
        self.vel = np.array(vel, dtype=float)
        self.acc_std = np.array(acc_noise, dtype=float)
        self.pos = np.array(pos, dtype=float)
        self.acc = np.array(acc, dtype=float)
        self.acc_update_interval = dt
        self.time_since_last_update = dt
        self.acc_diff = 0
        self.dt = dt

    def update(self):
        if self.time_since_last_update >= self.acc_update_interval:
            self.acc_diff = np.random.randn(*self.acc.shape) * self.acc_std
            self.time_since_last_update = 0
        else:
            self.time_since_last_update += self.dt

        self.acc += self.acc_diff * self.dt
        self.vel += self.acc * self.dt
        self.pos += self.vel * self.dt

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


def print_history(state_hist, pos_hist, vel_hist, acc_hist):
    for i in range(len(state_hist)):
        print("Time step: ", i)
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




def SecondOrderKF(R_std, Q, dt, P=100):
    """Create second order Kalman filter.
    Specify R and Q as floats."""

    tracker = KalmanFilter(dim_x=6, dim_z=2)

    tracker.F = np.array(
        [
            [1, dt, 0.5 * dt * dt, 0, 0, 0],
            [0, 1, dt, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, dt, 0.5 * dt * dt],
            [0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    tracker.u = 0.0
    tracker.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])

    tracker.R = np.eye(2) * R_std**2
    q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    tracker.x = np.array([[0, 0, 0, 0, 0, 0]]).T
    tracker.P = np.eye(6) * P
    return tracker


R_std = 1
Q_std = 2
dt = 1
next_t = 3 * dt

# simulate target movement
N = 300
sensor = Sensor((0, 0), (2, 0.2), acc_noise=(1,1), dt=dt)

zs = []
vels = []
accs = []
for i in range(N):
    sensor.update()
    zs.append(sensor.read_pos())
    vels.append(sensor.read_vel())
    accs.append(sensor.read_acc())

zs = np.array(zs)
vels = np.array(vels)
accs = np.array(accs)

# run filter
robot_tracker = SecondOrderKF(R_std=R_std, Q=Q_std, dt=dt)
mu, cov, _, _ = robot_tracker.batch_filter(zs)

state_hist = []

for x, P in zip(mu, cov):
    # covariance of x and y
    state_hist.append(x)
    cov = np.array([[P[0, 0], P[3, 0]], [P[0, 3], P[3, 3]]])
    mean = (x[0, 0], x[3, 0])
    x_pred = predict_position(x, next_t)
    plt.plot(x_pred[0, 0], x_pred[3, 0], marker="v", color="red")
    plot_covariance_ellipse(mean, cov=cov, fc="g", std=3, alpha=0.5)


print_history(state_hist, pos_hist=zs, vel_hist=vels, acc_hist=accs)

# plot results
plt.plot(mu[:, 0], mu[:, 3])
plt.plot(zs[:, 0], zs[:, 1])
plt.legend(loc=2)
plt.show()
