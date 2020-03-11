import os
import numpy as np
import pickle
from util.mnistutil import load_dataset
from networks.neural_network_squared_cost import NeuralNetworkSquared

def get_batch_mask(total_size, batch_size):
    return np.random.choice(total_size, batch_size)

def train_network(iterations, batch_size, save_network=False):
    # Load data
    (training_data, testing_data) = load_dataset(normalize=True, bitmapped=True)
    (train_images, train_solutions) = training_data
    (test_images, test_solutions) = testing_data

    # Create fresh Neutral Network
    network = NeuralNetworkSquared(layer_sizes=[784, 50, 10], learning_rate=1)

    total_size = train_images.shape[0]

    # Get random image batch
    batch_mask = get_batch_mask(total_size, batch_size)

    images_batch = train_images[batch_mask]
    solutions_batch = train_solutions[batch_mask]

    # Train
    for i in range(iterations):
        network.back_propogate(images_batch, solutions_batch)


    print("\n\n...\nTraining Complete\n")
