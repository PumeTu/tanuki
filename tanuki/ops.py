from .autograd import Ops, Node, Tensor, TensorOps
from typing import Union, Tuple

# This will be changed out later to a custom backend interface for GPU acceleration
import numpy as array_api
NDArray = array_api.ndarray

class Add(TensorOps):
    """Element-wise addition"""
    def forward(self, a: NDArray, b: NDArray):
        return a + b
    
    def backward(self, outgrad: "Node", node: "Node"):
        return outgrad, outgrad