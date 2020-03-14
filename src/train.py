
import argparse
import scripts.train_network as trainer


def main():
    parser = argparse.ArgumentParser(
        description='Configure and train a Neural Network')
    parser.add_argument('-i', '--iterations', type=int,
                        help='set number of training iterations. Defaults to 10000')
    parser.add_argument('-b', '--batchsize', type=int,
                        help='set number of current images processed during network training. Default to 500. Max value: 60000')
    parser.add_argument('-l', '--layers', type=int, nargs='+',
                        help='set hidden layer nueron counts. Defaults to 50')
    parser.add_argument('-s', '--save', action='store_true',
                        help='save network to disk. Defaults to False')

    args = parser.parse_args()

    iterations = 10000 if not args.iterations else args.iterations
    batch_size = 500 if not args.batchsize else args.batchsize
    hidden_layers = [50] if not args.layers else args.layers
    save_network = False if not args.save else args.save

    epoch_length = iterations // 10
    trainer.train_network(iterations=iterations,
                          batch_size=batch_size,
                          hidden_layers=hidden_layers,
                          epoch_length=120)


if __name__ == "__main__":
    main()
