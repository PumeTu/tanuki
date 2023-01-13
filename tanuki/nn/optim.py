import tanuki as tnk
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for param in self.params:
            param.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}

    def step(self):
        raise NotImplementedError()

class Adam(Optimizer):
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.m = {}
        self.u = {}

    def step(self):
        raise NotImplementedError()
