from Layers.Base import BaseLayer
import numpy as np

class TanH(BaseLayer):
    def __init__(self):
        super().init()
        self.out = None

    def forward(self, input_tensor):
        self.out = np.tanh(input_tensor)
        return self.out

    def backward(self, error_tensor):
        return error_tensor * (1 - pow(self.out, 2))