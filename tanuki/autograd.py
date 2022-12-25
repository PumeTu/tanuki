import tanuki
from typing import Optional, List, Union, Tuple, NamedTuple
from collections import namedtuple
import numpy

LAZY_MODE = False

import numpy as array_api
NDArray = array_api.ndarray

LAZY_MODE = False
class Ops:
    """Operation Base Class"""
    def __call__(self, *args):
        return NotImplementedError()

    def forward(self, *args: Tuple[NDArray]):
        """
        Calculates forward pass of operator
        
        Parameters
        ----------
        input: np.ndarray
            A list of input arrays

        Returns
        -------
        output: ndarray
            Array output of operation
        """
        raise NotImplementedError()

    def backward(self, outgrad: "Node", node: "Node") -> Union["Node", Tuple["Node"]]:
        """
        Computes the backward pass i.e. local derivative of the node

        Parameters
        ----------
        outgrad: Node
            gradient of parent node (chain rule)

        node: Node
            current node to compute the local gradient on

        Returns
        -------
        grad: Node or Tuple[Node]
            output gradient
        """
        raise NotImplementedError()

class Node:
    """A node in the computation graph to connect tensors together and hold relationships"""
    _op: Optional[Ops]
    _inputs: List["Node"]
    cahced_data: NDArray
    requires_grad:bool = False

    def realize_cached_data(self):
        """Run compute to realize data"""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self._op.forward(*[x.realize_cached_data() for x in self._inputs])
        return self.cached_data
    
    def is_leaf(self):
        return self._op is None

    def _init(
        self,
        _op: Optional[Ops],
        _inputs: List["Tensor"],
        cached_data: List[object] = None,
        requires_grad:bool = None 
    ):
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in _inputs)
        self._op = _op
        self._inputs = _inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

class Tensor(Node):
    """Subclass of Node that corresponds to the multi-dimensional array"""
    def __init__(
        self,
        array,
        device=None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        self._init(None, [], cached_data=cached_data, requires_grad=requires_grad)

    @staticmethod
    def _array_from_numpy(array, device, dtype):
        if array_api is numpy:
            return numpy.array(array, dtype=dtype)
        return array_api.array(array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(_op: Ops, _inputs: List["Node"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(_op, _inputs)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._int(None, [], data=data if not isinstance(data, Tensor) else data.realize_cached_data(), requires_grad=requires_grad)
        return tensor

    @property
    def data(self):
        return self.detach()
    
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, f'Invalid data type {value.dtype}, expecting {self.dtype}'
        self.data = value.realize_cached_data()

    def detach(self):
        """Creates a new tensor that shares the data but detaches from the graph"""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __repr__(self):
        return f'Tensor ({self.realize_cached_data()})'

    def __add__(self, other):
        return tanuki.ops.Add()(self, other)


class TensorOps(Ops):
    """Ops class that output tensors"""
    def __call__ (self, *args):
        return Tensor.make_from_op(self, args)