import numpy as np
import h5py

def sigmoid(x):
    """
    Sigmoid
    
    Arguments:
        x -- numpy array, inputs
    Returns:
        y -- numpy array, outpus with same shape as x
    """
    
    y = 1/(1+np.exp(-x))
    
    return y


def relu(x):
    """
    Rectified Linear Unit.

    Arguments:
        x -- numpy array, inputs

    Returns:
        y -- numpy array, outpus with same shape as x
    """
    
    y = np.maximum(0, x)
    assert(y.shape == x.shape)
    
    return y


def load_data():
    # load train data
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x = np.array(train_dataset["train_set_x"][:])
    train_set_y = np.array(train_dataset["train_set_y"][:])
    # load test data
    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    test_set_x = np.array(test_dataset["test_set_x"][:])
    test_set_y = np.array(test_dataset["test_set_y"][:])
    # load class
    classes = np.array(test_dataset["list_classes"][:])
    # reshape labels
    train_set_y = train_set_y.reshape((-1, 1))
    test_set_y = test_set_y.reshape((-1, 1))

    return train_set_x, train_set_y, test_set_x, test_set_y, classes
