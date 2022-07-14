import numpy as np
from .Base import BaseLayer
from .TanH import TanH
from .Sigmoid import Sigmoid
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    _optimizer: object = None
    _gradient_weights = 0
    _memorize = False

    def __init__(self, input_size, hidden_size, output_size):
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.trainable = True
        self.input_size = input_size  # the dimension of the input vector
        self.hidden_size = hidden_size  # the dimension of the hidden state
        self.output_size = output_size

        self.hidden_fc_layer = FullyConnected(input_size + hidden_size, hidden_size)
         # Before initializing weights it will initialize randomly
        self.output_fc_layer = FullyConnected(hidden_size, output_size)

        self.hidden_state = np.zeros((1, self.hidden_size))
        self.output_fc_layer_gradient_weights = np.zeros_like(self.output_fc_layer.weights)

    def initialize(self, weights_initializer, bias_initializer):
        self.hidden_fc_layer.initialize(weights_initializer, bias_initializer)
        self.output_fc_layer.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        # input 13 * 9
        # batch 9
        # hidden_state 7
        # output 5
        batch_size = input_tensor.shape[0]
        self.hidden_fc_layer_input_tensor = np.zeros((batch_size, self.hidden_size + self.input_size + 1))
        self.output_fc_layer_input_tensor = np.zeros((batch_size, self.hidden_size + 1))
        self.tan_activations = np.zeros((batch_size, self.hidden_size))
        self.sigmoid_activations = np.zeros((batch_size, self.output_size))

        output_tensor = np.zeros((batch_size, self.output_size))  # initialize hidden_state

        if not self._memorize:
            hidden_state = np.zeros((1, self.hidden_size))  # initialize hidden_state with zero
        else:
            hidden_state = self.hidden_state

        for t in range(batch_size):
            # concatenation previous hidden state with corresponding input tensor
            x_hat = np.concatenate((hidden_state, input_tensor[t, None]), axis=1)  # 1*21

            tan_input = self.hidden_fc_layer.forward(x_hat)

            # line below: x_hat, saved for set as input_tensor of hidden layer in the backward mode
            self.hidden_fc_layer_input_tensor[t] = self.hidden_fc_layer.input_tensor

            # update hidden state, should I save it?
            hidden_state = self.tanh.forward(tan_input)  # 1*7

            # saved for set activations in backward mode
            self.tan_activations[t] = self.tanh.tanh_output

            # output
            transition_of_hy = self.output_fc_layer.forward(hidden_state)
            # hidden_state with bias, saved for set as input_tensor of output layer in backward
            self.output_fc_layer_input_tensor[t] = self.output_fc_layer.input_tensor

            output_tensor[t] = self.sigmoid.forward(transition_of_hy)
            #  saved for set activations in backward mode
            self.sigmoid_activations[t] = self.sigmoid.sigmoid_output

        # save hidden state in the last iteration for initialize hidden state for the next sequence
        self.hidden_state = self.tan_activations[-1, None]
        return output_tensor

    def backward(self, error_tensor):
        time_step = error_tensor.shape[0] - 1
        output_tensor = np.zeros((error_tensor.shape[0], self.input_size))
        gradient_previous_hidden_state = np.zeros((1, self.hidden_size))
        self.gradient_weights = np.zeros_like(self.hidden_fc_layer.weights)

        while time_step >= 0:  # can use np.flip(error_tensor, axis=0)
            self.sigmoid.sigmoid_output = self.sigmoid_activations[time_step, None]
            sigmoid_error = self.sigmoid.backward(error_tensor[time_step, None])  # 1*5

            self.output_fc_layer.input_tensor = self.output_fc_layer_input_tensor[time_step, None]
            # hidden layer at time step t, 7*1
            output_fc_layer_error = self.output_fc_layer.backward(sigmoid_error)

            # self.output_fc_layer_weights += self.output_fc_layer.weights
            self.output_fc_layer_gradient_weights += self.output_fc_layer.gradient_weights
            #  copy backpropagation
            self.tanh.tanh_output = self.tan_activations[time_step, None]
            tanh_error = self.tanh.backward(gradient_previous_hidden_state + output_fc_layer_error)

            self.hidden_fc_layer.input_tensor = self.hidden_fc_layer_input_tensor[time_step, None]
            hidden_fc_layer_error = self.hidden_fc_layer.backward(tanh_error)

            self.gradient_weights += self.hidden_fc_layer.gradient_weights

            gradient_previous_hidden_state = hidden_fc_layer_error[:, :self.hidden_size]
            gradient_with_res_to_input = hidden_fc_layer_error[:, self.hidden_size:]
            output_tensor[time_step] = gradient_with_res_to_input

            # calculation the gradiant
            time_step -= 1
        if self.optimizer is not None:
            self.output_fc_layer.weights = self.optimizer.calculate_update(
                self.output_fc_layer.weights, self.output_fc_layer_gradient_weights)
            self.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights)
        return output_tensor

    def calculate_regularization_loss(self):
        if self._optimizer.regularizer:
            self.regularization_loss += self._optimizer.regularizer.norm(self.hidden_fc_layer.weights)
            self.regularization_loss += self._optimizer.regularizer.norm(self.output_fc_layer.weights)
        return self.regularization_loss

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, x):
        self._memorize = x

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x

    @property
    def weights(self):
        return self.hidden_fc_layer.weights

    @weights.setter
    def weights(self, x):
        self.hidden_fc_layer.weights = x

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, x):
        self._gradient_weights = x
