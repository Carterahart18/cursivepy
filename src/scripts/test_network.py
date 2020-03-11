import os
import numpy as np
import pickle
from util.mnist import load_dataset
from networks.neural_network_v3 import NeuralNetworkV3

def get_batch_mask(total_size, batch_size):
    return np.random.choice(total_size, batch_size)

def test_network():
    pass
