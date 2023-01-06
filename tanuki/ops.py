from .autograd import Op, Node, Tensor, TensorOp
from typing import Union, Tuple, Optional

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

def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)

class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def backward(self, outgrad: Tensor, node: Tensor):
        return outgrad * self.scalar * power_scalar(node.inputs[0], self.scalar-1)

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)

class EWiseDiv(TensorOp):
    def forward(self, a: NDArray, b: NDArray):
        return a / b

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def div(a, b):
    return EWiseDiv()(a, b)

class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def forward(self, a: NDArray):
        return a / self.scalar

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def div_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a: NDArray):
        return array_api.transpose(a, axes=self.axes)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def transpose(a, axes=None):
    return Transpose(axes)(a)

class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a: NDArray):
        return array_api.reshape(a, self.shape)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def reshape(a, shape):
    return Reshape(shape)(a)

class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, a: NDArray):
        return array_api.broadcast_to(a, self.shape)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)

class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def forward(self, a: NDArray):
        return array_api.sum(a, axis=self.axes)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def summation(a, axes=None):
    return Summation(axes)(a)

class MatMul(TensorOp):
    def forward(self, a: NDArray, b: NDArray):
        return array_api.matmul(a, b)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def matmul(a, b):
    return MatMul()(a, b)

class Negate(TensorOp):
    def forward(self, a: NDArray):
        return array_api.negative(a)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def negate(a):
    return Negate()(a)

class Log(TensorOp):
    def forward(self, a: NDArray):
        return array_api.log(a)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def log(a):
    return Log()(a)

class Exp(TensorOp):
    def forward(self, a: NDArray):
        return array_api.exp(a)

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def exp(a):
    return Exp()(a)

class ReLU(TensorOp):
    def forward(self, a: NDArray):
        raise NotImplementedError()

    def backward(self, outgrad: Tensor, node: Tensor):
        raise NotImplementedError()

def relu(a):
    return ReLU()(a)