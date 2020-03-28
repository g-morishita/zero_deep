import numpy as np


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


t = np.array([0, 0, 1])
y = np.array([0.1, 0.3, 0.6])
print(f"MSE: {mean_squared_error(y, t)}")
print(f"CE: {cross_entropy_error(y, t)}")
