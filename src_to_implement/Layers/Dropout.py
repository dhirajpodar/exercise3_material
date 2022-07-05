from Layers.Base import BaseLayer

import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, input_tensor):

        if self.testing_phase:
            return input_tensor

        self.mask = np.random.rand(*input_tensor.shape) < self.probability
        output = np.multiply(input_tensor, self.mask)
        output /= self.probability
        return output

    def backward(self, output_tensor):
        return np.multiply(output_tensor, self.mask)/self.probability

