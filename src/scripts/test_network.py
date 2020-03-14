import numpy as np
from util.mnist import load_dataset
from networks.neural_network import NeuralNetwork


def get_batch_mask(total_size, batch_size):
    return np.random.choice(total_size, batch_size)


def test_network(iterations, batch_size, network):
    # Load testing data
    (training_data, testing_data) = load_dataset(normalize=True, bitmapped=True)
    (test_images, test_solutions) = testing_data

    total_size = test_images.shape[0]

    print("  Testing Accuracy  ")
    print("--------------------")

    for i in range(iterations):
        # Get random image / solution batch pair
        batch_mask = get_batch_mask(total_size, batch_size)
        images_batch = test_images[batch_mask]
        solutions_batch = test_solutions[batch_mask]

        # Train network
        test_acc = network.accuracy(images_batch, solutions_batch) * 100
        print(f"       {test_acc:05.2f}%")
