import tanuki
from typing import Optional, List, Union, Tuple, NamedTuple, Dict
from collections import namedtuple
import numpy

LAZY_MODE = False
TENSOR_COUNTER = 0

import numpy as array_api
NDArray = array_api.ndarray

class Device:
    """Indicates the device supporting NDArray"""

class CPUDevice(Device):
    """"""
    def __repr(self):
        return "tanuki.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True
    
    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)
    
    def randn(self, *shape):
        """
        Note: numpy doesn't support types within standard random routines,
                and .astype("float32") does work if we're generating a singleton
        """
        return numpy.random.randn(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

def cpu():
    return CPUDevice()

def all_devices():
    return [cpu()]


class Op:
    """Operation Base Class"""
    def __call__(self, *args):
        return NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """
        Calculates forward pass of operator
        
        Args:
            input (ndarray): A list of input arrays

        Returns:
        output (ndarray): Array output of operation
        """
        raise NotImplementedError()

    def gradient(self, outgrad: "Node", node: "Node") -> Union["Node", Tuple["Node"]]:
        """
        Computes the backward pass i.e. local derivative of the node

        Args:
        outgrad (Node): gradient of parent node (chain rule)

        node (Node): current node to compute the local gradient on

        Returns:
            grad (Node or Tuple[Node]): output gradient
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, outgrad: "Node", node: "Node") -> Tuple["Node"]:
        """
        Convenient method to return tuple from gradient call
        """
        output = self.gradient(outgrad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class TensorOps(Op):
    """Op class that ouputs tensors"""
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""
    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)

class Node:
    """A node in the computation graph to connect tensors together and hold relationships"""
    op: Optional[Op]
    inputs: List["Node"]
    cahced_data: NDArray
    requires_grad:bool = False

    def realize_cached_data(self):
        """Run compute to realize data"""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data
    
    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        cached_data: List[object] = None,
        requires_grad:bool = None 
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(None, [], cached_data=data, requires_grad=requires_grad)
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Node"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class TensorTuple(Node):
    """Represent a tuple of tensors.
    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return tanuki.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "tanuki.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return tanuki.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())

class Tensor(Node):
    """Subclass of Node that corresponds to the multi-dimensional array"""
    grad: "Tensor"

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
    def make_from_op(op: Op, inputs: List["Node"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
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

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else Tensor(numpy.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __repr__(self):
        return f'Tensor ({self.realize_cached_data()})'

    def __add__(self, other):
        if isinstance(other, Tensor):
            return tanuki.ops.EWiseAdd()(self, other)
        else:
            return tanuki.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return tanuki.ops.EWiseMul()(self, other)
        else:
            return tanuki.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return tanuki.ops.PowerScalar(other)(self)
    
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return tanuki.ops.EWiseAdd()(self, tanuki.ops.Negate()(other))
        else:
            return tanuki.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return tanuki.ops.EWiseDiv()(self, other)
        else:
            return tanuki.ops.DivScalar(other)(self)
    
    def __matmul__(self, other):
        return tanuki.ops.MatMul()(self, other)

    def matmul(self, other):
        return tanuki.ops.MatMul()(self, other)
        
    def sum(self, axes=None):
        return tanuki.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return tanuki.ops.BroadcastTo(shape)(self)
    
    def reshape(self, shape):
        return tanuki.ops.Reshape(shape)(self)

    def __neg__(self):
        return tanuki.ops.Negate()(self)

    def transpose(self, axes=None):
        return tanuki.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmult__ = __matmul__

def compute_gradient_of_variables(output_tensor, outgrad):
    """
    Computes gradient of output node with respect to each node in node list then stores the computed 
        gradient in the grad field of each variable
    """
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [outgrad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        grad = sum_node_list(node_to_output_grads_list[node])
        node.grad = grad
        if node.op:
            grad_tuple = node.op.gradient_as_tuple(grad, node)
            for i, input in enumerate(node.inputs):
                node_to_output_grads_list[input] = node_to_output_grads_list.get(input, [])
                node_to_output_grads_list[input].append(grad_tuple[i])
    return node_to_output_grads_list
        
def find_topo_sort(node_list: List[Node]) -> List[Node]:
    """
    Given a list of nodes, return a topological sort list of nodes ending in them
    """
    topo_order = []
    visited = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """
    Post Order DFS
    """
    visited.append(node)
    for input in node.inputs:
        if input not in visited:
            topo_sort_dfs(input, visited, topo_order)
    topo_order.append(node)

class TensorOp(Op):
    """Op class that output tensors"""
    def __call__ (self, *args):
        return Tensor.make_from_op(self, args)

##############################
####### Helper Methods #######
##############################

def sum_node_list(node_list):
    """
    Custom sum function in order to avoid creating redundant nodes in Python sum implementation
    """
    from operator import add
    from functools import reduce

    return reduce(add, node_list)