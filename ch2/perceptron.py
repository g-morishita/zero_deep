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


# implementation of AND using perceptron with bias term
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


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = w @ x + b
    if tmp <= 0:
        return 0
    else:
        return 1


# check if NAND works properly
assert NAND(0, 0) == 1, 'expected 1, but output is {NAND(0, 0)}'
assert NAND(1, 0) == 1, 'expected 1, but output is {NAND(1, 0)}'
assert NAND(0, 1) == 1, 'expected 1, but output is {NAND(0, 1)}'
assert NAND(1, 1) == 0, 'expected 0, but output is {NAND(1, 1)}'


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.3, 0.3])
    b = -0.2
    tmp = w @ x + b
    if (tmp <= 0):
        return 0
    else:
        return 1


# check if OR works properly
assert OR(0, 0) == 0, 'expected 1, but output is {OR(0, 0)}'
assert OR(1, 0) == 1, 'expected 1, but output is {OR(1, 0)}'
assert OR(0, 1) == 1, 'expected 1, but output is {OR(0, 1)}'
assert OR(1, 1) == 1, 'expected 0, but output is {OR(1, 1)}'


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


# check if XOR works properly
assert XOR(0, 0) == 0, 'expected 0, but output is {XOR(0, 0)}'
assert XOR(1, 0) == 1, 'expected 1, but output is {XOR(1, 0)}'
assert XOR(0, 1) == 1, 'expected 1, but output is {XOR(0, 1)}'
assert XOR(1, 1) == 0, 'expected 0, but output is {XOR(1, 1)}'
