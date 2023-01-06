import sys
sys.path.append('./')
import tanuki as tnk
import unittest
import numpy as np

#------------------------------------------------------------------------------------------------
# Test Forward Pass

class TestOpsFoward(unittest.TestCase):
    def test_add(self):
        a, b = tnk.Tensor([1.2, 3.2, 15.3]), tnk.Tensor([3.3, 4.32, 2.3])
        a_np, b_np = np.array([1.2, 3.2, 15.3]), np.array([3.3, 4.32, 2.3]) 
        np.testing.assert_allclose(tnk.add(a, b).numpy(), a_np + b_np)

    def test_add_scalar(self):
        a, a_np = tnk.Tensor([1.2, 3.2, 15.3]), np.array([1.2, 3.2, 15.3])
        scalar = 2
        np.testing.assert_allclose(tnk.add_scalar(a, scalar).numpy(), a_np + scalar)

    def test_mul(self):
        a, b = tnk.Tensor([1.2, 3.2, 15.3]), tnk.Tensor([3.3, 4.32, 2.3])
        a_np, b_np = np.array([1.2, 3.2, 15.3]), np.array([3.3, 4.32, 2.3]) 
        np.testing.assert_allclose(tnk.mul(a, b).numpy(), np.multiply(a_np, b_np))

    def test_mul_scalar(self):
        a, a_np = tnk.Tensor([1.2, 3.2, 15.3]), np.array([1.2, 3.2, 15.3])
        scalar = 2
        np.testing.assert_allclose(tnk.mul_scalar(a, scalar).numpy(), a_np * scalar)
   
    def test_power_scalar(self):
        a, a_np = tnk.Tensor([1.2, 3.2, 15.3]), np.array([1.2, 3.2, 15.3])
        scalar = 2
        np.testing.assert_allclose(tnk.power_scalar(a, scalar).numpy(), np.power(a_np, scalar))

    def test_div(self):
        pass

    def test_div_scalar(self):
        pass

    def test_matmul(self):
        pass

    def test_summation(self):
        pass

    def test_broadcast_to(self):
        pass

    def test_reshape(self):
        pass

    def test_negate(self):
        pass

    def test_transpose(self):
        pass
    
#------------------------------------------------------------------------------------------------
# Test Backward Pass
def gradient_check(f, *args, eps=1e-6, backward=False, **kwargs):
    pass

class TestOpsBackward(unittest.TestCase):
    def test_add(self):
        pass

    def test_add_scalar(self):
        pass

    def test_mul(self):
       pass

    def test_mul_scalar(self):
       pass

    def test_power_scalar(self):
        pass
    
    def test_div(self):
        pass

    def test_div_scalar(self):
        pass

    def test_matmul(self):
        pass

    def test_summation(self):
        pass

    def test_broadcast_to(self):
        pass

    def test_reshape(self):
        pass

    def test_negate(self):
        pass

    def test_transpose(self):
        pass

if __name__ == '__main__':
    unittest.main()