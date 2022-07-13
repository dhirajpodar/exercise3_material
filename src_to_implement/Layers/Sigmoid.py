from Layers.Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.sigmoid_output = None

    def forward(self, input_tensor):
        self.sigmoid_output = 1 / (1 + np.exp(-input_tensor))
        return self.sigmoid_output

    def backward(self, error_tensor):
        return error_tensor * self.sigmoid_output * (1 - self.sigmoid_output)
