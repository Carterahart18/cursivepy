import numpy
import pickle
from util.functions import sigmoid, sigmoid_derivative, softmax, cross_entropy_error


class NeuralNetwork():
    """
    NeuralNetwork

    Data format
    -----------
    * The "layers" field of the class is a list of layers 0 to N
    * Each layer is a dictionary with members "weights", "biases", and "size"
        * "weights" is an 2D numpy array
            * Each row represents one neuron in the current layer
            * Each column represent the weights from the current neuron to each neruon
                in the next layer
            * Example: If layer i has 5 neurons and layer j has 3 neurons, weights for
              layer 1 would look like this:
                    [[w11, w12, w13]
                     [w21, w22, w23]
                     [w31, w32, w33]
                     [w41, w42, w43]
                     [w51, w52, w53]]
              where w(x,y) represents the weight from neuron x in layer i to nueron y
              in layer i + 1

        * "biases" is an 1D numpy array
            * Each row represent the current layer's bias toward each neuron in the next
              layer
            * Example: For a given layer i mapping to layer i + 1 who has 5 neurons, the
              biases array looks like this:
                    [b1, b2, b3, b4, b5]
              where b(x) represents the bias added to neuron x in layer i + 1
    """

    def __init__(self, layer_sizes, weight_init_std=0.01, learning_rate=1):
        self.layers = []
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            # For each node in layer i, create a weight for each neuron in layer i + 1
            layer_weights = weight_init_std * \
                numpy.random.randn(layer_sizes[i], layer_sizes[i+1])
            # Create a bias for each destination neuron
            layer_biases = numpy.zeros(layer_sizes[i+1])
            self.layers.append({
                "weights": layer_weights,
                "biases": layer_biases,
                "size": layer_sizes[i]
            })

    def set_layers(self, layers):
        self.layers = layers

    def _convertOutputToResult(self, output_batch):
        """
        Parameters
        ----------
        output_batch: An batch of outputs, each an array representing the final output
        layer of the neural network

        Returns
        -------
        A batch of results, represented by the index of the neuron most strongly activated
        """
        return numpy.argmax(output_batch, axis=1)

    def _predict(self, input_batch):
        """
        Parameters
        ----------
        input_batch: An batch of inputs

        Returns
        -------
        A tuple of the form (predictions_batch, layer_activations_batch_raw, layer_activations_batch_sig)
        """

        batch_size = input_batch.shape[0]

        layer_activations_batch_raw = [input_batch]
        layer_activations_batch_sig = [sigmoid(input_batch)]
        layer_activations_batch = input_batch

        for i in range(len(self.layers)):
            layer = self.layers[i]

            weight_sum_batch = numpy.dot(
                layer_activations_batch, layer['weights'])

            # Process Layer L + 1 activations
            layer_output_batch = weight_sum_batch + layer['biases']
            layer_activations_batch_raw.append(layer_output_batch)

            # Apply sigmoid to Layer L + 1 activations
            layer_output_batch = sigmoid(layer_output_batch)
            layer_activations_batch_sig.append(layer_output_batch)

            # Layer L + 1 inputs are this layer's outputs
            layer_activations_batch = layer_output_batch

        predictions_batch = layer_output_batch
        return (predictions_batch, layer_activations_batch_raw, layer_activations_batch_sig)

    def predict(self, input_batch):
        (predictions_batch,
            layer_activations_batch_raw,
            layer_activations_batch_sig) = self._predict(input_batch)
        return predictions_batch

    def loss(self, input_batch, solution_batch):
        """
        Parameters
        ----------
        input_batch: A batch of inputs (layer 1 activations)
        solution_batch: A batch of expected outputs (output layer activations)

        Returns
        -------
        An loss amount for each image in the input using the Cross Entropy Error function
        """
        predictions = self.predict(input_batch)
        return cross_entropy_error(predictions, solution_batch)

    def accuracy(self, input_batch, solution_batch):
        batch_size = input_batch.shape[0]

        predictions = self.predict(input_batch)

        predictions = self._convertOutputToResult(predictions)
        solution_batch = self._convertOutputToResult(solution_batch)

        accuracy = numpy.sum(predictions == solution_batch) / float(batch_size)

        return accuracy

    def _cost(self, actual_batch, expected_batch):
        batch_size = actual_batch.shape[0]
        return (actual_batch - expected_batch) / batch_size

    def train(self, input_batch, solution_batch):
        batch_size = input_batch.shape[0]

        (predictions_batch,
         layer_activations_batch_raw,
         layer_activations_batch_sig) = self._predict(input_batch)

        error_batch = self._cost(predictions_batch, solution_batch)
        layer_gradients = list()

        # Iterates backwards through layers, comparing Layer i to the error at Layer i + 1
        # to calculate the weight and bias gradients per layer
        for i in range(len(self.layers) - 1, -1, -1):
            weight_gradient = numpy.dot(
                layer_activations_batch_sig[i].T, error_batch)
            bias_gradient = numpy.sum(error_batch, axis=0)

            layer_gradients.insert(0, {
                "weights": weight_gradient,
                "biases": bias_gradient,
            })

            # Calculate error for current layer
            layer_weights = self.layers[i]['weights']
            error_batch = sigmoid_derivative(
                layer_activations_batch_raw[i]) * numpy.dot(error_batch, layer_weights.T)

        # Apply gradients to each layer
        for i in range(len(self.layers)):
            self.layers[i]['weights'] -= self.learning_rate * \
                layer_gradients[i]['weights']

            self.layers[i]['biases'] -= self.learning_rate * \
                layer_gradients[i]['biases']

    def save_network(self, dir_path, file_name):
        file_path = dir_path + "/" + file_name
        try:
            f = open(file_path, 'wb')
            pickle.dump(self.layers, f, -1)
            print("Successfully saved network to", file_path)
        except IOError:
            print("Failed to save network to", file_path)

    @staticmethod
    def load_network(dir_path, file_name):
        file_path = dir_path + "/" + file_name
        try:
            f = open(file_path, 'rb')
            layers = pickle.load(f)
            print("Successfully loaded network from", file_path)

            network = NeuralNetwork(layer_sizes=[0])
            network.set_layers(layers)
            return network
        except IOError:
            print("Failed to load network from", file_path)
