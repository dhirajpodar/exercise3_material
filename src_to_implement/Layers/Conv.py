import numpy as np
from .Base import BaseLayer
import scipy.signal as sc


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, (num_kernels))
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = None
        self._optimizer = None
        self._bias_optimizer = None


    def padding_calculation(self, x):
        if x % 2 != 0:
            return (x // 2, x // 2)
        else:
            return (x // 2, x // 2 - 1)

    def forward(self, input_tensor):
        self.original_input_tensor = np.copy(input_tensor)
        # Get the value of parameters of Input Tensor based on input length
        if len(input_tensor.shape) == 4:
            N, C, row, col = input_tensor.shape
            WW = self.convolution_shape[2]
            self.stride_width = self.stride_shape[1]
        else:
            N, C, row = input_tensor.shape
            WW = self.convolution_shape[1]
            self.stride_width = self.stride_shape[0]
        # No. of filters
        self.stride_height = self.stride_shape[0]
        self.padSize_w = (0, 0)
        inp_pad_w = 0

        self.pad_h = self.padding_calculation(self.convolution_shape[1])
        Height_Out = int(((row - self.convolution_shape[1] + sum(self.pad_h)) / self.stride_height) + 1)
        self.stride_index = []

        # if the convolution shape is 1D
        if len(self.convolution_shape) == 2:
            self.input_tensor = np.zeros((N, C, sum(self.pad_h) + row))
            output_tensor = np.zeros((N, self.num_kernels, Height_Out))
            self.stride_index = [i for i in range(0, row, self.stride_height)]
            for b in range(N):
                for c in range(C):
                    self.input_tensor[b, c] = np.pad(input_tensor[b, c], self.pad_h, mode='constant')

        # if the convolution shape is 2D
        else:
            self.padSize_w = self.padding_calculation(WW)
            inp_pad_w = sum(self.padSize_w) + col
            self.input_tensor = np.zeros((N, C, sum(self.pad_h) + row, inp_pad_w))
            weight_out = int(((col - WW + sum(self.padSize_w)) / self.stride_width) + 1)
            output_tensor = np.zeros((N, self.num_kernels, Height_Out, weight_out * (len(self.convolution_shape) > 2)))
            iter = 0
            for i in range(0, col, self.stride_width):
                iter = i * row
                for j in range(0, row, self.stride_height):
                    self.stride_index.append(iter)
                    iter += self.stride_height
            for b in range(N):
                for c in range(C):
                    self.input_tensor[b, c] = np.pad(input_tensor[b, c], (self.pad_h, self.padSize_w), mode='constant')

        # query through each batch
        for n in range(N):
            # query through filters
            for f in range(self.num_kernels):
                array_c = []
                for c in range(self.weights.shape[1]):
                    pad = sc.correlate(input_tensor[n, c], self.weights[f, c], mode='same')
                    array_c.append(pad)
                array_c = np.stack(array_c, axis=0).sum(axis=0).flatten()[self.stride_index].reshape(
                    output_tensor.shape[2:])
                output_tensor[n, f] = array_c + self.bias[f]
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = np.zeros_like(self.original_input_tensor)
        out_w = np.copy(self.weights)
        N = error_tensor.shape[0]
        err_N, err_C = error_tensor.shape[0], error_tensor.shape[1]
        grad_w = np.zeros((error_tensor.shape[0], *out_w.shape))

        for b in range(err_N):
            for c in range(err_C):
                count = 0
                curr_error = error_tensor[b, c].flatten()
                temp_error = np.zeros((self.original_input_tensor.shape[2:])).flatten()
                for ind in self.stride_index:
                    temp_error[ind] = curr_error[count]
                    count += 1
                temp_error = temp_error.reshape(self.original_input_tensor.shape[2:])
                for c_out in range(self.original_input_tensor.shape[1]):
                    grad_w[b, c, c_out] = sc.correlate(self.input_tensor[b, c_out], temp_error, mode='valid')
        self.gradient_weights = grad_w.sum(axis=0)

        # gradient calc with respect to input
        if len(self.convolution_shape) == 3:
            output_weights = np.transpose(out_w, (1, 0, 2, 3))
        else:
            output_weights = np.transpose(out_w, (1, 0, 2))
        output_tensor = np.zeros_like(self.original_input_tensor)

        for b in range(N):
            for c in range(output_weights.shape[0]):
                c_out = []
                for k in range(output_weights.shape[1]):
                    i = 0
                    curr_error = error_tensor[b, k].flatten()
                    temp_error = np.zeros((self.original_input_tensor.shape[2:])).flatten()
                    for index in self.stride_index:
                        temp_error[index] = curr_error[i]
                        i += 1
                    temp_error = temp_error.reshape(self.original_input_tensor.shape[2:])
                    temp_conv = sc.convolve(temp_error, output_weights[c, k], mode='same')
                    c_out.append(temp_conv)
                output_tensor[b, c] = np.stack(c_out, axis=0).sum(axis=0)

        # gradient calc with respect to bias
        if len(self.convolution_shape) == 3:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2))
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        return output_tensor

    def initialize(self, weights_initializer, bias_initializer):
        conv_channel = self.convolution_shape[0]
        kernel_height = self.convolution_shape[1]
        kernel_width = self.convolution_shape[2]

        if len(self.convolution_shape) == 3:
            self.weights = weights_initializer.initialize((self.num_kernels, conv_channel, kernel_height, kernel_width),
                                                          conv_channel * kernel_height * kernel_width,
                                                          self.num_kernels * kernel_height * kernel_width)
            self.bias = bias_initializer.initialize((self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

        elif len(self.convolution_shape) == 2:
            self.weights = weights_initializer.initialize((self.num_kernels, conv_channel, kernel_height),
                                                          conv_channel * kernel_height, self.num_kernels * kernel_height)
            self.bias = bias_initializer.initialize((1, self.num_kernels), 1, self.num_kernels)
            self.bias = self.bias[-1]

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
