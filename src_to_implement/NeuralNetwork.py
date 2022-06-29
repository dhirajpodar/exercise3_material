import numpy as np
from Optimization.Optimizers import *
import copy


class NeuralNetwork:
    def __init__(self, optimizers, weights_initializer, bias_initializer):
        self.optimizers = optimizers
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.label_tensor = None
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        output_layer = np.copy(input_tensor)
        for layer in self.layers:
            output_layer = layer.forward(output_layer)
        return self.loss_layer.forward(output_layer, self.label_tensor)

    def backward(self):
        output_back = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            output_back = layer.backward(output_back)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizers)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def phase(self):
        pass