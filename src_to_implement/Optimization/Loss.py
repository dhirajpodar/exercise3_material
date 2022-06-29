import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None
        self.epsilon = np.finfo(float).eps

    def forward(self, prediction_tensor, label_tensor):
        self.input_tensor = prediction_tensor

        '''epsilon the smallest representation number, increases stability for every
        wrong predictions to prevent values close to log(0)'''
        #epsilon = np.finfo(float).eps

        loss = -np.sum(label_tensor * np.log(prediction_tensor + self.epsilon)) # loss calculation
        return loss

    def backward(self, label_tensor):
        return -(label_tensor / self.input_tensor+self.epsilon)
