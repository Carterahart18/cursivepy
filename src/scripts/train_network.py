import os
import numpy as np
import pickle
from util.mnist import load_dataset
from networks.neural_network import NeuralNetwork

DIST_PATH = os.path.dirname(os.path.abspath(__file__)) + '/../../dist'
DIST_PATH = os.path.realpath(DIST_PATH)


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
    network = NeuralNetwork(layer_sizes=layer_sizes)

    total_size = train_images.shape[0]

    if batch_size > total_size:
        raise Exception("Batch size exceeds total training examples")

    print("  Training Accuracy  |  Testing Accuracy  |  Percent Complete  ")
    print("---------------------+--------------------+--------------------")

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
            percent_complete = i / iterations * 100
            print(f"       {train_acc:05.2f}%        |", end="")
            print(f"       {test_acc:05.2f}%       |", end="")
            print(f"       {percent_complete:05.2f}%       ")

    print("\nTraining Complete !\n")

    if save_network:
        network.save_network(DIST_PATH + "/network.pkl")
    else:
        answer = None
        answers = ["y", "yes", "n", "no"]
        while answer not in answers:
            answer = input("Would you like to save this network (y/n)? ")
            answer = answer.lower()
            if answer not in answers:
                print("Please response (y/n).")
        if answer == "y" or answer == "yes":
            network.save_network(DIST_PATH + "/network.pkl")
