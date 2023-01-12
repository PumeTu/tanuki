import struct
import gzip
import numpy as np
import tanuki as tnk

def parse_mnist(image_path, label_path):
    """
    Read images and albels file in MNIST format
    Args:
        image_path (str): name of gzipped images file in MNIST format
        label_path (str): name of gzipped labels file in MNIST format

    Returns:
        X (np.ndarray[float32]): 2D numpy array containing loaded data. The dimensionality should be
            (num_examples, input_dim) where input_dim is the full dimension of the data, e.g. MNIST images 
            are 28x28 thus input_dim = 784. Values should be of type np.float32 and normalized
        y (np.ndarray[int8]): 1D numpy array containing labels of the examples. Values should be of type
            np.int8 and for MNIST contain values 0-9.
    """
    with gzip.open(image_path) as f:
        X = np.frombuffer(f.read(), "B", offset=16).reshape(-1, 784).astype(np.float32) / 255.
    with gzip.open(label_path) as f:
        y = np.frombuffer(f.read(), "B", offset=8).astype(np.int8)
    return X, y


def cross_entropy_loss(logits, y_one_hot):
    """
    Cross Entropy Loss
    Args:
        logits (tnk.Tensor[float32]): 2D logit prediction Tensor for each class of shape (batch_size, num_classes)
        y_one_hot (tnk.Tensor[int8]): 2D one hot encoded Tensor of shape(batch_size, num_classes)

    Returns:
        loss (tnk.Tensor[float32]): Average softmax loss over the sample
    """
    return tnk.summation(tnk.log(tnk.summation(tnk.exp(logits), axes=1)) - tnk.summation(logits * y_one_hot, axes=1) / logits.shape[0])

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """
    Run a single epoch of SGD for a two-layer neural network defined by W1 and W2 (weights) with no bias
    Args:
        X (np.ndarray[float32]): 2D input array of size (num_examples, input_dim)
        y (np.ndarray[int8]): 1D class label array of size (num_examples,)
        W1 (tnk.Tensor[float32]): 2D array of first layer weights of shape (input_dim, hidden_dim)
        W2 (tnk.Tensor[float32]): 2D array of second layer weights of shape (hidden_dim, output_dim)
        lr (float): learning rate
        batch (int): size of SGD mini-batch
    
    Returns:
        W1 (tnk.Tensor[float32]): trained weight of first layer
        W2 (tnk.Tensor[float32]): trained weight of second layer
    """
    raise NotImplementedError()