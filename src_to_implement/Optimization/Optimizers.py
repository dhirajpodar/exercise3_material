import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.momentum_rate * self.vk - self.learning_rate * gradient_tensor
        if self.regularizer:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return weight_tensor + self.vk



class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.vk = self.rk = self.bias_correlation_vk = self.bias_correlation_rk = 0
        self.epsilon = np.finfo(float).eps

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.vk = self.mu * self.vk + (1 - self.mu) * gradient_tensor
        self.rk = self.rho * self.rk + (1 - self.rho) * (gradient_tensor * gradient_tensor)
        self.bias_correlation_vk = self.vk / (1 - self.mu ** self.k)
        self.bias_correlation_rk = self.rk / (1 - self.rho ** self.k)
        self.k += 1

        if self.regularizer:
            weight_tensor -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        weight_tensor -= self.learning_rate * (
                    self.bias_correlation_vk / (np.sqrt(self.bias_correlation_rk) + self.epsilon))
        return weight_tensor
