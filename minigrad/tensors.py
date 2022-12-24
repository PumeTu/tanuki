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



    
        

