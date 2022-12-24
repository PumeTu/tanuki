import numpy as np

class Tensor:
    '''
    
    '''
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0. if requires_grad else None
        self._backward = lambda: None

    def zero_grad(self):
        self.grad = 0. if self.requires_grad else None

    def __repr__(self):
        return f'Tensor({self.data}, requires_grad = {self.requires_grad}'

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.out = Tensor(self.data + other.data)

        return self.out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        self.out = Tensor(self.data - other.data)

        return self.out

    
        

