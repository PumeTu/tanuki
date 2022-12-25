import numpy as np
from typing import Optional, List
from ops import Operations

class Node:
    '''
    A single Node in the computational graph
    Args:
        data
        requires_grad
        _inputs
        _op
    '''
    def __init__(self, 
                data: np.ndarray,
                _inputs: List["Node"],
                _op: Optional[Operations],
                requires_grad: bool = False
    ):
        self.data = data
        self.grad = 0 if requires_grad else None
        self._inputs = set(_inputs)
        self._op = _op
        self.requires_grad = requires_grad
