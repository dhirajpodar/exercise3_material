from Layers.Base import BaseLayer
from Layers import Helpers
import numpy as np
import copy


class BatchNormalization(BaseLayer):

    def __init__(self, channels):
        super().__init__()
        self.x_hat = None
        self.trainable = True
        self.testing_phase = False
        self.channels = channels
        self.bias = np.zeros(channels)
        self.weights = np.ones(channels)
        self._gradient_bias = None
        self._gradient_weights = None
        self._optimizer = None  # weight optimizer
        self._bias_optimizer = None
        self.moving_mean = 0
        self.moving_var = 1
        self.mean = 0
        self.variance = 1
        self.epsilon = 1e-15

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        alpha = 0.8

        # Mean and variance of Batch
        if len(input_tensor.shape) == 2:
            self.mean = np.mean(input_tensor, axis=0)
            self.variance = np.var(input_tensor, axis=0)

            if self.testing_phase:
                x_hat = (input_tensor - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
            else:
                batch_mean = np.mean(input_tensor, axis=0)
                batch_var = np.var(input_tensor, axis=0)

                self.moving_mean = (alpha * self.mean) + (1 - alpha) * batch_mean
                self.moving_var = (alpha * self.variance) + (1 - alpha) * batch_var

                self.mean = batch_mean
                self.var = batch_var

                x_hat = (input_tensor - self.mean) / (np.sqrt(self.var + self.epsilon))
            self.x_hat = x_hat
            y_hat = self.weights * x_hat + self.bias

        if len(input_tensor.shape) == 4:  # If it is convolution

            # reshape the B x H x M x N tensor to B x H x M . N
            new_input_tensor = self.reformat(input_tensor)
            self.mean = np.mean(new_input_tensor, axis=0)
            self.variance = np.var(new_input_tensor, axis=0)

            if self.testing_phase:
                x_hat = (new_input_tensor - self.moving_mean) / np.sqrt(self.moving_var + self.epsilon)
            else:
                batch_mean = np.mean(new_input_tensor, axis=0)
                batch_var = np.var(new_input_tensor, axis=0)

                self.moving_mean = (alpha * self.mean) + (1 - alpha) * batch_mean
                self.moving_var = (alpha * self.variance) + (1 - alpha) * batch_var

                self.mean = batch_mean
                self.var = batch_var

                x_hat = (new_input_tensor - self.mean) / (np.sqrt(self.var + self.epsilon))

            # we need to reverse it
            # from B . M . N x H to B x M . N x H with reshape
            _x_hat = self.reformat(x_hat)
            self.x_hat = _x_hat

            y_hat = self.weights.reshape(1, input_tensor.shape[1], 1, 1) * _x_hat + \
                    self.bias.reshape(1, input_tensor.shape[1], 1, 1)

        return y_hat

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

            self.gradient_weights = np.sum(error_tensor * self.x_hat, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)

        elif len(error_tensor.shape) == 4:
            y_hat = Helpers.compute_bn_gradients(self.reformat(error_tensor),
                                                 self.reformat(self.input_tensor),
                                                 self.weights,
                                                 self.mean,
                                                 self.variance,
                                                 self.epsilon)
            y_hat = self.reformat(y_hat)
            self.gradient_weights = np.sum(error_tensor * self.x_hat, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return y_hat

    def initialize(self, weights_initializer, bias_initializer):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)

    def reformat(self, tensor):

        output = np.zeros_like(tensor)
        new_tensor = copy.deepcopy(tensor)

        if len(tensor.shape) == 2:
            output = new_tensor.reshape(self.input_tensor.shape[0], -1, self.input_tensor.shape[1])

            # from B x M . N x H to B x H x M . N with transpose
            output = output.transpose(0, 2, 1)

            # from B x H x M . N to B x H x M x N  with reshape
            output = output.reshape(self.input_tensor.shape[0],
                                    self.input_tensor.shape[1],
                                    self.input_tensor.shape[2],
                                    self.input_tensor.shape[3])

        if len(tensor.shape) == 4:
            output = new_tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])

            # transpose from B x H x M . N to B x M . N x H
            output = output.transpose(0, 2, 1)

            # reshape again to have a B . M . N x H tensor
            output = output.reshape(-1, new_tensor.shape[1])

        return output

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value
