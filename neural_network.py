import numpy
from functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error


class NeuralNetwork:
    """
    NeuralNetwork

    A simple class to train against solution data
    """

    def __init__(self, layer_sizes, weight_init_std=0.01, learning_rate=1):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            # For each node in layer i, create a weight for each neuron in
            # layer i + 1
            layer_weights = weight_init_std * \
                numpy.random.randn(layer_sizes[i], layer_sizes[i+1])
            # Create a bias for each destination neuron
            layer_biases = numpy.zeros(layer_sizes[i+1])
            self.layers.append({
                "weights": layer_weights,
                "biases": layer_biases,
                "size": layer_sizes[i]
            })

    def _convertOutputToResult(self, output):
        return numpy.argmax(output, axis=1)

    def predict(self, images):
        """
        Parameters
        ----------
        images: An array of images represented by an array of 784 elements from 0 to 1

        Returns
        -------
        An array of predictions for each image
        """

        layer_input = images

        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_output = numpy.dot(
                layer_input, layer['weights']) + layer['biases']

            if i < len(self.layers) - 1:
                layer_output = sigmoid(layer_output)
            else:
                layer_output = softmax(layer_output)

            layer_input = layer_output

        predictions = layer_output
        return predictions

    def loss(self, images, solutions):
        """
        Parameters
        ----------
        images: An array of images represented by an array of 784 elements from 0 to 1
        solutions: An array of image solutions presented by a bitmap for 0 through 9

        Returns
        -------
        An loss amount for each image in the input using the Cross Entropy Error function
        """
        predictions = self.predict(images)
        return cross_entropy_error(predictions, solutions)

    def accuracy(self, images, solutions):
        predictions = self.predict(images)

        predictions = self._convertOutputToResult(predictions)
        solutions = self._convertOutputToResult(solutions)

        accuracy = numpy.sum(predictions == solutions) / float(images.shape[0])
        return accuracy

    def back_propogate(self, images, solutions):
        number_of_images = len(images)

        layer_input = images
        raw_outputs = []
        outputs = []

        # Iterate forward through the layers
        for i in range(len(self.layers)):
            layer = self.layers[i]
            layer_output = numpy.dot(
                layer_input, layer['weights']) + layer['biases']
            raw_outputs.append(layer_output)

            if i < len(self.layers) - 1:
                layer_output = sigmoid(layer_output)
            else:
                layer_output = softmax(layer_output)

            outputs.append(layer_output)
            layer_input = layer_output

        predictions = layer_output

        diff = (predictions - solutions) / number_of_images
        gradients = []

        # Iterate backwards through the layers
        for i in range(len(self.layers) - 1, -1, -1):
            if i != 0:
                layer_output = outputs[i - 1]
            else:
                layer_output = images

            current_weights = self.layers[i]['weights']

            layer_weight_grads = numpy.dot(layer_output.T, diff)
            layer_bias_grads = numpy.sum(diff, axis=0)

            gradients.insert(0, {
                "weights": layer_weight_grads,
                "biases": layer_bias_grads,
            })

            if i != 0:
                diff = sigmoid_grad(
                    raw_outputs[i - 1]) * numpy.dot(diff, current_weights.T)

        for i in range(len(self.layers)):
            self.layers[i]['weights'] -= self.learning_rate * \
                gradients[i]['weights']

            self.layers[i]['biases'] -= self.learning_rate * \
                gradients[i]['biases']
