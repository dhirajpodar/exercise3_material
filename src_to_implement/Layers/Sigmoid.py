from Layers.Base import BaseLayer
import numpy as np

class Sigmoid(BaseLayer):
    def __init__(self):
        super().init()
        self.output = None

    def forward(self, input_tensor):
        self.output = (1/1-np.exp(-input_tensor))

    def backward(self, error_tensor):
        return error_tensor * self.output * (1 - self.output)
