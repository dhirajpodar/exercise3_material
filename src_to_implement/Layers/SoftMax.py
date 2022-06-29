from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.y_prediction = None
        self.error_tensor = None

    def forward(self, input_size):
        exp_output = np.exp(input_size - np.max(input_size, axis=1, keepdims=True))
        self.y_prediction = exp_output / np.sum(exp_output, axis=1, keepdims=True)

        return self.y_prediction

    def backward(self, error_tensor):
        softmax_gradient = self.y_prediction * (error_tensor - np.sum(error_tensor*self.y_prediction, axis=1, keepdims=True))
        return softmax_gradient
