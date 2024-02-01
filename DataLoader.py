import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    fnames = os.listdir(data_dir)
    x_train = np.array([[]]).reshape(0,3072)
    y_train = np.array([])
    for fn in fnames:
        
        if not (fn.startswith("data") or fn.startswith("test")):
            continue
        
        with open(os.path.join(data_dir,fn), 'rb') as fo:
            ds = pickle.load(fo, encoding='bytes')
            xtemp = np.array(ds[b'data']) 
            ytemp = np.array(ds[b'labels'])

        if fn.startswith("test_batch"):
            x_test = xtemp
            y_test = ytemp

        if fn.startswith("data_batch"):
            x_train = np.concatenate((xtemp,x_train), axis=0)
            y_train = np.concatenate((ytemp,y_train), axis=0)

    ### YOUR CODE HERE
    print(x_train.shape, y_train.shape)
    return x_train, y_train, x_test, y_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    train_split_index = int(x_train.shape[0] * train_ratio)
    x_train_new = x_train[:train_split_index]
    y_train_new = y_train[:train_split_index]

    x_valid = x_train[train_split_index:]
    y_valid = y_train[train_split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

def load_testing_images(privatedir):
   
    # private_dir = os.path.join(privatedir, "Private")
    x_private_test = np.load(os.path.join(privatedir, 'private_test_images_2022.npy'))
    print("private data set shape",x_private_test.shape)
    return x_private_test
