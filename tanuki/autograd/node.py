import numpy as np
from typing import Optional, List
from autograd.ops import Operations

class Node:
    """A node in the computation graph to connect tensors together and hold relationships"""
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

    def data_compute(self):
        """Run compute on data"""
        if self.data is not None:
            return self.data
        self.data = self._op.compute(*[x.data_compute() for x in self._inputs])
        return self.data
    
