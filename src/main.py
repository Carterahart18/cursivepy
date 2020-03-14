import os
import argparse
from ui.app import App
from networks.neural_network import NeuralNetwork

DEFAULT_FILE = '../dist/network.pkl'
DEFAULT_LEARNING_RATE = 0.05

CURR_DIR = os.path.dirname(os.path.abspath(__file__)) + '/'


def main():
    parser = argparse.ArgumentParser(
        description='Drawing program to test a Neural Network')
    parser.add_argument('-f', '--file', type=str, nargs='?',
                        help='location of network to load. Defaults to ' +
                        DEFAULT_FILE)
    parser.add_argument('-l', '--learning', type=float, nargs='?',
                        help='learning rate for the network. Defaults to ' +
                        str(DEFAULT_LEARNING_RATE))

    args = parser.parse_args()

    file_path = CURR_DIR
    file_path += DEFAULT_FILE if not args.file else args.file
    file_path = os.path.realpath(file_path)
    learning_rate = DEFAULT_LEARNING_RATE if not args.learning else args.learning

    try:
        network = NeuralNetwork.load_network(file_path)
    except IOError:
        exit()

    app = App(network=network, learning_rate=learning_rate)
    app.start()


if __name__ == '__main__':
    main()
