from Layers.Base import BaseLayer
import numpy as np


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.tanh_output = None

    def forward(self, input_tensor):
        self.tanh_output = np.tanh(input_tensor)
        return self.tanh_output

    def backward(self, error_tensor):
        return error_tensor * (1 - pow(self.tanh_output, 2))
