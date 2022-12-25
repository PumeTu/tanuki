import sys
sys.path.append('./')
import tanuki as tnk
import unittest
import numpy as np

class TestOps(unittest.TestCase):
    
    def test_add(self):
        a = tnk.Tensor([1, 2, 3])
        b = tnk.Tensor([3, 4, 5])
        c = a + b
        na = np.array([1, 2, 3])
        nb = np.array([3, 4, 5])
        nc = na + nb
        np.testing.assert_allclose(c.numpy(), nc)

if __name__ == '__main__':
    unittest.main()