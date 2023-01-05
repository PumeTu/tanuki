from autograd import Ops, Node, Tensor, TensorOp
from typing import Union, Tuple

# This will be changed out later to a custom backend interface for GPU acceleration
import numpy as array_api
NDArray = array_api.ndarray

class EWiseAdd(TensorOp):
    def forward(self, a: NDArray, b: NDArray):
        return a + b
    
    def backward(self, outgrad: Tensor, node: Tensor):
        return outgrad, outgrad

def add(a, b):
    return EWiseAdd()(a, b)

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a + self.scalar

    def backward(self, outgrad: Tensor, node: Tensor):
        return outgrad

def add_scalar(a, scalar):
    return AddScalar(scalar)(a)

class EWiseMul(TensorOp):
    def forward(self, a: NDArray, b: NDArray):
        return a * b

    def backward(self, outgrad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return outgrad * rhs, outgrad * lhs

def mul(a, b):
    return EWiseMul()(a, b)

class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a * self.scalar

    def backward(self, outgrad: Tensor, node: Tensor):
        return (outgrad * self.scalar,)

def mul_scalr(a, scalar):
    return MulScalar(scalar)(a)

class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray) -> NDArray:
        raise NotImplementedError()

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWiseDiv(TensorOp):
    def forward(self, a: NDArray, b: NDArray):
        raise NotImplementedError()

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def div(a, b):
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def foward(self, a: NDArray):
        return a / self.scalar

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def div_scalar(a, scalar):
    return DivScalar(scalar)(a)