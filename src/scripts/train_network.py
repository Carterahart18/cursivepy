import os
import numpy as np
import pickle
from util.mnist import load_dataset
from networks.neural_network import NeuralNetwork

dist_path = os.path.dirname(os.path.abspath(__file__)) + '/../../dist'


def get_batch_mask(total_size, batch_size):
    return np.random.choice(total_size, batch_size)


def train_network(iterations, batch_size, epoch_length, hidden_layers=[50], save_network=False):

    if batch_size < 1:
        raise Exception("Batch size must be at least 1")

    # Load data
    (training_data, testing_data) = load_dataset(normalize=True, bitmapped=True)
    (train_images, train_solutions) = training_data
    (test_images, test_solutions) = testing_data

    # Create fresh Neutral Network
    layer_sizes = [784, *hidden_layers, 10]
    print(layer_sizes)
    network = NeuralNetwork(layer_sizes=layer_sizes)

    total_size = train_images.shape[0]

    if batch_size > total_size:
        raise Exception("Batch size exceeds total training examples")

    print("  Training Accuracy  |  Testing Accuracy  ")
    print("---------------------+--------------------")

    for i in range(iterations):
        # Get random image / solution batch pair
        batch_mask = get_batch_mask(total_size, batch_size)
        images_batch = train_images[batch_mask]
        solutions_batch = train_solutions[batch_mask]

        # Train network
        network.train(images_batch, solutions_batch)

        # Print out results every ${epoch_length} iterations
        if i % epoch_length == 0:
            train_acc = network.accuracy(train_images, train_solutions) * 100
            test_acc = network.accuracy(test_images, test_solutions) * 100
            print(f"        {train_acc:05.2f}%       ", end="")
            print("|", end="")
            print(f"        {test_acc:05.2f}%       ")

    print("\n\n...\nTraining Complete\n")

    network.save_network(dist_path + "/network.pkl")
