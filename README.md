# CursivePy

## About

"CursivePy" is a simple project that implements a Neural Network and teaches it to recognize simple handwritten digits. You can configure the network as you please with custom hidden layers, training batch sizes and iterations, and save and load network connections. Your network can then be tested against your own handwriting using a drawing program which simulates.

The 28 x 28 handwritten digit recognition problem is often considered a "hello world" program for neural networks. It is a simple, well defined challenged that can be easily solved by a via a trained Neural Network but very difficult to solve with conventional discrete programming methods.

## Training the network

To create a configuration of the network, run the following:

```shell
python src/train.py
```

This script will automatically download the training and testing data from [MNSIT](https://en.wikipedia.org/wiki/MNIST_database) and train the network to decipher the images in a dry run using the default configurations.

To save the network for use in testing, add the `-s` flag.

To configure the number of testing iterations use `-i <num>`.

To configure the image batch size of each iteration use `-b <num>`.

To configure the hidden layers of the network use `-l [layer_size...]`.

For more information, run `python src/train.py -h`.

## Testing the network

Once a network has been saved to disk, you can run the `src/main.py` script to open a GUI that allows you to draw your own custom characters!


## TODO

* Fill out setup, installation, and learning, challenges, motivations, learnings and testing README
* Connect Paint program to trained neural network
* Create a wrapper UI around the paint program
* (Optional) Create a wrapper UI around the Machine Learning
