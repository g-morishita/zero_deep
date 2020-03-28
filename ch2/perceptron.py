import numpy as np


# implementation of AND using perceptron
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    else:
        return 1

# check if AND works properly
assert AND(0, 0) == 0, 'expected 0, but output is {AND(0, 0)}'
assert AND(1, 0) == 0, 'expected 0, but output is {AND(1, 0)}'
assert AND(0, 1) == 0, 'expected 0, but output is {AND(0, 1)}'
assert AND(1, 1) == 1, 'expected 1, but output is {AND(1, 1)}'


def AND_bias(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = w @ x + b
    if tmp <= 0:
        return 0
    else:
        return 1


# check if AND_bias works properly
assert AND_bias(0, 0) == 0, 'expected 0, but output is {AND_bias(0, 0)}'
assert AND_bias(1, 0) == 0, 'expected 0, but output is {AND_bias(1, 0)}'
assert AND_bias(0, 1) == 0, 'expected 0, but output is {AND_bias(0, 1)}'
assert AND_bias(1, 1) == 1, 'expected 1, but output is {AND_bias(1, 1)}'
