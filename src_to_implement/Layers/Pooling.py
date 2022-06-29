from Layers.Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.batch_size = None
        self.channel_size = None
        self.stride_row = None
        self.stride_col = None
        self.pool_row = None
        self.pool_col = None
        self.row_path_size = None
        self.col_path_size = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        padding = 0  # 'valid padding'
        self.batch_size = input_tensor.shape[0]
        self.channel_size = input_tensor.shape[1]
        self.stride_row = self.stride_shape[0]
        self.stride_col = self.stride_shape[1]
        self.pool_row = self.pooling_shape[0]
        self.pool_col = self.pooling_shape[1]
        self.row_path_size = int((input_tensor.shape[2] - self.pool_row + 2 * padding) / self.stride_row) + 1
        self.col_path_size = int((input_tensor.shape[3] - self.pool_col + 2 * padding) / self.stride_col) + 1
        output = np.zeros((self.batch_size, self.channel_size, self.row_path_size, self.col_path_size))

        for b in range(self.batch_size):
            for c in range(self.channel_size):
                for i in range(0, self.row_path_size):
                    for j in range(0, self.col_path_size):
                        row_start = i * self.stride_row
                        row_end = row_start + self.pool_row
                        col_start = j * self.stride_col
                        col_end = col_start + self.pool_col

                        # Use the corners to define the current slice on the ith training example of input_tensor,
                        # channel c.
                        input_slice = input_tensor[b, c, row_start:row_end, col_start:col_end]

                        # Compute the pooling operation on the slice. Use an if statement to differentiate the modes.
                        # Use np.max.
                        output[b, c, i, j] = np.max(input_slice)
        return output

    def backward(self, error_tensor):
        back_output = np.zeros(self.input_tensor.shape)
        for b in range(self.batch_size):
            a_prev = self.input_tensor[b]
            for c in range(self.channel_size):
                for i in range(0, self.row_path_size):
                    for j in range(0, self.col_path_size):
                        row_start = i * self.stride_row
                        row_end = row_start + self.pool_row
                        col_start = j * self.stride_col
                        col_end = col_start + self.pool_col

                        prev_slice = a_prev[c, row_start:row_end, col_start:col_end]
                        mask = (prev_slice == np.max(prev_slice))

                        back_output[b, c, row_start:row_end, col_start:col_end] += np.multiply(mask, error_tensor[b, c, i, j])

        return back_output


