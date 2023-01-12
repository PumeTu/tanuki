import sys
sys.path.append('./apps')
from apps.simple_ml import cross_entropy_loss, parse_mnist
import numpy as np
import tanuki as tnk
import unittest
from test.test_ops import gradient_check

class TestSimpleML(unittest.TestCase):
    def test_softmax_loss(self):
        # test forward pass for log
        np.testing.assert_allclose(tnk.log(tnk.Tensor([[4.  ],
        [4.55]])).numpy(), np.array([[1.38629436112 ],
        [1.515127232963]]))

        # test backward pass for log
        gradient_check(tnk.log, tnk.Tensor(1 + np.random.rand(5,4)))
        X,y = parse_mnist("/home/pumetu/data/mnist/train-images-idx3-ubyte.gz",
                        "/home/pumetu/data/mnist/train-labels-idx1-ubyte.gz")
        np.random.seed(0)
        Z = tnk.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
        y_one_hot = np.zeros((y.shape[0], 10))
        y_one_hot[np.arange(y.size), y] = 1
        y = tnk.Tensor(y_one_hot)
        np.testing.assert_allclose(cross_entropy_loss(Z,y).numpy(), 2.3025850, rtol=1e-6, atol=1e-6)
        Z = tnk.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
        np.testing.assert_allclose(cross_entropy_loss(Z,y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6)

        # test softmax loss backward
        Zsmall = tnk.Tensor(np.random.randn(16, 10).astype(np.float32))
        ysmall = tnk.Tensor(y_one_hot[:16])
        gradient_check(cross_entropy_loss, Zsmall, ysmall, tol=0.01, backward=True)
