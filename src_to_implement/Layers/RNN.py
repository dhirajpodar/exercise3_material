class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False

    def forward(self, input_tensor):
        pass

    def backward(self, output_tensor):
        pass

    def initialize(self, weights_initializer, bias_initializer):
        pass