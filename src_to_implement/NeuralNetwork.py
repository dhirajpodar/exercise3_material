
from copy import deepcopy

import numpy as np


class NeuralNetwork:
    testing_phase = bool
    optimizer = ''
    regularization_loss = 0
    loss = []
    layers = []
    data_layer: np.ndarray(shape=None, dtype=np.float64)
    loss_layer = None
    input_tensor: np.ndarray(shape=None, dtype=np.float64)
    label_tensor: np.ndarray(shape=None, dtype=np.float64)
    weights_initializer: np.ndarray(shape=None, dtype=np.float64)
    bias_initializer: np.ndarray(shape=None, dtype=np.float64)

    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = deepcopy(optimizer)
        self.weights_initializer = deepcopy(weights_initializer)
        self.bias_initializer = deepcopy(bias_initializer)

    def __del__(self):
        self.layers.clear()

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = input_tensor.astype('float64')
        self.label_tensor = self.label_tensor.astype('float64')
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer.regularizer:
                self.regularization_loss += layer.optimizer.regularizer.norm(layer.weights)  # λ||w||2
        if self.loss_layer is not None:
            input_tensor = self.loss_layer.forward(input_tensor, self.label_tensor)
            input_tensor += self.regularization_loss
        return input_tensor

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations: int):
        self.phase = False
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        act_input = input_tensor
        for layer in self.layers:
            layer.testing_phase = True
            act_input = layer.forward(act_input)
        return act_input

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, var):
        self.testing_phase = var
