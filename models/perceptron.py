import numpy as np


class Perceptron:
    def __init__(self, n, eta=0.1):
        self.dim = n
        self.w = np.random.uniform(size=n)
        self.eta = eta
