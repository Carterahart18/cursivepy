import os
import argparse
import ui.window as UI
from networks.neural_network import NeuralNetwork

dist_path = os.path.dirname(os.path.abspath(__file__)) + '/../dist'


def main():
    parser = argparse.ArgumentParser(
        description='Drawing program to test a Neural Network')
    parser.add_argument('-f', '--file', type=str, nargs='?',
                        help='location of network to load. Defaults to /dist/network.pkl')

    args = parser.parse_args()

    file_path = dist_path + "/"
    file_path += "network.pkl" if not args.file else args.file

    network = NeuralNetwork.load_network(file_path)
    app = UI.Window(network=network)
    app.start()


if __name__ == "__main__":
    main()
