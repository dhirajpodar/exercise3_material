from Layers.Base import BaseLayer

import numpy as np


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)  # bias is also included in the weights
        self._optimizer = None
        self.gradient = None
        self.input_tensor = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = np.concatenate((input_tensor, np.ones((input_tensor.shape[0], 1))), axis=1)

        output = np.dot(self.input_tensor, self.weights)
        return output

    def backward(self, error_tensor):
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        self.error_tensor = np.delete(self.error_tensor, -1, axis=1)

        self.gradient = np.dot(self.input_tensor.T, error_tensor)
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient)
        return self.error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((self.weights, bias))

    # getter method
    @property
    def optimizer(self):
        return self._optimizer

    # setter method
    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x

    # getter method
    @property
    def gradient_weights(self):
        return self.gradient

    # setter method
    @gradient_weights.setter
    def gradient_weights(self, x):
        self.gradient = x
