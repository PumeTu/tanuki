from typing import List, Callable, Any
from tanuki.autograd import Tensor
from tanuki import ops
import tanuki.nn.init as init
import numpy as np

class Parameter(Tensor):
    """Special kind of Tensor for Modules class"""

def _get_params(value):
    if isinstance(value, Parameter):
        return [value]
    if isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _get_params(v)
        return params
    if isinstance(value, Module):
        return value.parameters()


class Module:
    def parameters(self):
        return _get_params(self.__dict__)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)