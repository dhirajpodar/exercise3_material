from Layers.Base import BaseLayer
from Layers import Helpers
import numpy as np
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.input_tensor = None
        self.epsilon = np.finfo(float).eps
        self.weights = np.zeros(channels)
        self.bias = np.ones(channels)
        self.mean = 0
        self.variance = 1

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        Y_hat = np.zeros(input_tensor.shape)
        alpha = 0.8
        self.mean = np.mean(input_tensor, axis=0)
        self.variance = np.var(input_tensor, axis=0)
        batch_mean = np.mean(input_tensor, axis=0)
        batch_var = np.var(input_tensor, axis=0)
        moving_mean = (alpha * self.mean) + (1 - alpha) * batch_mean
        moving_var = (alpha * self.variance) + (1 - alpha) * batch_var
        if len(input_tensor) == 3:

            if self.testing_phase:
                mean = moving_mean
                variance = moving_var
            else:
                mean = batch_mean
                variance = batch_var

            X_hat = (input_tensor - mean) / (np.sqrt(np.pow(variance, 2) + self.epsilon))
            Y_hat = self.weights * X_hat + self.bias

        if len(input_tensor) == 4:  # if  it is convolution
            # reshape the B x H x M x N tensor to B x H x M . N
            new_input_tensor = self.reformat(input_tensor)

            if not self.testing_phase:
                mean = batch_mean
                variance = batch_var
            else:
                mean = moving_mean
                variance = moving_var

            X_hat = (new_input_tensor - mean) / (np.sqrt(variance + self.epsilon))
            X_hat = self.reformat(X_hat)
            Y_hat = self.weights.reshape(1, input_tensor.shape[1], 1, 1) * X_hat + \
                    self.bias.reshape(1, input_tensor.shape[1], 1, 1)

        return Y_hat

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 2:

            ## gradient with respect to the input
            y_hat = Helpers.compute_bn_gradients(error_tensor,
                                                 self.input_tensor,
                                                 self.weights,
                                                 self.mean,
                                                 self.variance,
                                                 self.epsilon)

            ## gradient with respect to the weights and bias
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)


        elif len(error_tensor.shape) == 4:
            y_hat = Helpers.compute_bn_gradients(self.reformat(error_tensor),
                                                 self.reformat(self.input_tensor),
                                                 self.weights,
                                                 self.mean,
                                                 self.variance,
                                                 self.epsilon)
            y_hat = self.reformat(y_hat)
            self.gradient_weights = np.sum(error_tensor * self.X_hat, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return y_hat

    def reformat(self, tensor):

        output = np.zeros_like(tensor)
        new_tensor = copy.deepcopy(tensor)

        if len(tensor.shape) == 2:
            output = new_tensor.reshape(self.input_tensor.shape[0], -1, self.input_tensor.shape[1])

            # from B x M . N x H to B x H x M . N with transpose
            output = output.transpose(0, 2, 1)

            # from B x H x M . N to B x H x M x N  with reshape
            output = output.reshape(self.input_tensor.shape[0], self.input_tensor.shape[1], self.input_tensor.shape[2]
                                    , self.input_tensor.shape[3])

        if len(tensor.shape) == 4:
            output = new_tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])

            # transpose from B x H x M . N to B x M . N x H
            output = output.transpose(0, 2, 1)

            # reshape again to have a B . M . N x H tensor
            output = output.reshape(-1, new_tensor.shape[1])

        return output
