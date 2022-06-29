import numpy as np
from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.shape = []

    def forward(self, input_tensor):
        self.shape = input_tensor.shape
        return input_tensor.reshape(self.shape[0], int((np.prod(self.shape))/self.shape[0]))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
