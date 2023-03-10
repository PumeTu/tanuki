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
    loss = -tnk.summation(logits * y_one_hot, axes=1) + tnk.log(tnk.summation(tnk.exp(logits), axes=1))
    return tnk.summation(loss) / loss.shape[0]

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
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    for i in range(0, num_examples, batch):
        if i + batch > num_examples:
            x_batch = tnk.Tensor(X[i:num_examples])
            y_batch = y[i:num_examples]
        else:
            x_batch = tnk.Tensor(X[i:i+batch])
            y_batch = y[i:i+batch]
        logits = tnk.relu(x_batch @ W1) @ W2
        y_one_hot = np.zeros((y_batch.shape[0], num_classes), dtype=np.float32)
        y_one_hot[np.arange(y_batch.shape[0]), y_batch] = 1
        y_one_hot = tnk.Tensor(y_one_hot)
        loss = cross_entropy_loss(logits, y_one_hot)
        loss.backward()
        W1 = tnk.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = tnk.Tensor(W2.numpy() - lr * W2.grad.numpy())
    return W1, W2

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = tnk.Tensor(y_one_hot)
    return cross_entropy_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)