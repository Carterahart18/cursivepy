
import os
import argparse
import scripts.test_network as tester
from networks.neural_network import NeuralNetwork

DEFAULT_FILE = 'dist/network.pkl'
DEFAULT_ITERATIONS = 100
DEFAULT_BATCH = 500

CURR_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"


def main():
    parser = argparse.ArgumentParser(
        description='Drawing program to test a Neural Network')
    parser.add_argument('-f', '--file', type=str, nargs='?',
                        help='location of network to load. Defaults to ' + DEFAULT_FILE)
    parser.add_argument('-i', '--iterations', type=int,
                        help='set number of training iterations. Defaults to '
                        + str(DEFAULT_ITERATIONS))
    parser.add_argument('-b', '--batchsize', type=int,
                        help='set number of current images processed during network ' +
                        'training. Default to ' + str(DEFAULT_BATCH) +
                        '. Max value: 60000')

    args = parser.parse_args()

    iterations = DEFAULT_ITERATIONS if not args.iterations else args.iterations
    batch_size = DEFAULT_BATCH if not args.batchsize else args.batchsize
    file_path = CURR_DIR + '../'
    file_path += DEFAULT_FILE if not args.file else args.file

    try:
        network = NeuralNetwork.load_network(file_path)
    except IOError:
        exit()

    epoch_length = iterations // 30
    tester.test_network(iterations=iterations,
                        batch_size=batch_size,
                        network=network)


if __name__ == "__main__":
    main()
