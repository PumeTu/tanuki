import math
import tanuki as tnk

def rand(*shape, low=0.0, hight=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random numbers uniformly between low and high"""
    device = tnk.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return tnk.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

