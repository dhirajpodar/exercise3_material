from Layers.Base import BaseLayer

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass
