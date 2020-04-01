import os, sys
sys.path.append(os.path.pardir)
import numpy as np
from two_layer_net2 import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
    
    # if t is one-hot vector, t is reduced to label
    if y.size == t.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return - np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class MultiLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out 

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = x @ self.W + self.b

        return out

    def backward(self, dout):
        dx = dout @ self.W.T
        self.dW = self.x.T @ dout
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

if __name__ == '__main__':
    apple = 100
    num_apples = 2
    orange = 150
    num_oranges = 3
    tax = 1.1

    # layer
    mul_apple_layer = MultiLayer()
    mul_orange_layer = MultiLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MultiLayer()

    # forward
    apple_price = mul_apple_layer.forward(apple, num_apples)
    orange_price = mul_orange_layer.forward(orange, num_oranges)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    final_price = mul_tax_layer.forward(all_price, tax)

    print(f"The total is {final_price}")

    # backward
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dorange, dnum_oranges = mul_orange_layer.backward(dorange_price)
    dapple, dnum_apples = mul_apple_layer.backward(dapple_price)

    print(dnum_apples, dapple, dnum_oranges, dorange, dtax)
