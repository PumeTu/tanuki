import sys
sys.path.append('./apps')
from apps.simple_ml import cross_entropy_loss, parse_mnist, nn_epoch, loss_err
import numpy as np
import tanuki as tnk
import unittest
from test.test_ops import gradient_check
import numdifftools as nd

class TestSimpleML(unittest.TestCase):
    def test_cross_entropy_loss(self):
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
        gradient_check(cross_entropy_loss, Zsmall, ysmall, tolerance=0.01, backward=True)

    def test_nn_epoch(self):
        np.random.seed(0)
        X = np.random.randn(50,5).astype(np.float32)
        y = np.random.randint(3, size=(50,)).astype(np.uint8)
        W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
        W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
        W1_0, W2_0 = W1.copy(), W2.copy()
        W1 = tnk.Tensor(W1)
        W2 = tnk.Tensor(W2)
        X_ = tnk.Tensor(X)
        y_one_hot = np.zeros((y.shape[0], 3))
        y_one_hot[np.arange(y.size), y] = 1
        y_ = tnk.Tensor(y_one_hot)
        dW1 = nd.Gradient(lambda W1_ :
            cross_entropy_loss(tnk.relu(X_@tnk.Tensor(W1_).reshape((5,10)))@W2, y_).numpy())(W1.numpy())
        dW2 = nd.Gradient(lambda W2_ :
            cross_entropy_loss(tnk.relu(X_@W1)@tnk.Tensor(W2_).reshape((10,3)), y_).numpy())(W2.numpy())
        W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
        np.testing.assert_allclose(dW1.reshape(5,10), W1_0-W1.numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(dW2.reshape(10,3), W2_0-W2.numpy(), rtol=1e-4, atol=1e-4)

        # test full epoch
        X,y = parse_mnist("/home/pumetu/data/mnist/train-images-idx3-ubyte.gz",
                        "/home/pumetu/data/mnist/train-labels-idx1-ubyte.gz")
        np.random.seed(0)
        W1 = tnk.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
        W2 = tnk.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
        W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
        np.testing.assert_allclose(np.linalg.norm(W1.numpy()), 28.437788,
                                rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(W2.numpy()), 10.455095,
                                rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(loss_err(tnk.relu(tnk.Tensor(X)@W1)@W2, y),
                                (0.19770025, 0.06006667), rtol=1e-4, atol=1e-4)