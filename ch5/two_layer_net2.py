import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from layers import *

class TwoLayerNet:
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            output_size, 
            weight_init_std=0.01
            ):

        # initilization of weight
        self.params = {}
        self.params['W1'] = \
                weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = \
                weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # produce layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
                Affine(self.params['W1'], self.params['b1'])
        self.layers['Reule1'] = Relu()
        self.layers['Affine2'] = \
                Affine(self.params['W2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum( y == t) / x.shape[0]
        return accuracy

    def _numerical_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)

        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x) # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2 * h)

            x[idx] = tmp_val
            it.iternext()

        return grad

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
