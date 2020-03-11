import numpy
from util.functions import sigmoid, sigmoid_grad, softmax, cross_entropy_error


class NeuralNetworkSquared():
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

    def _convertOutputToResult(self, output):
        return numpy.argmax(output, axis=1)

    def predict(self, input_batch):
        """
        Parameters
        ----------
        input_batch: An array of input represented by an array of 784 elements from 0 to 1

        Returns
        -------
        An array of predictions for each image
        """

        # The entire process is calculated for a batch of inputs. This just means that
        # We're doing everything the same, but the inputs and outputs are arrays of size
        # batch_size, each representing one input's journey through the network

        # Set layer 1's values
        layer_input_batch = input_batch

        # For each layer
        for i in range(len(self.layers)):
            layer = self.layers[i]

            # The basic equation for each neuron y in layer i + 1 is the following:
            #   layer_i+1(y) = SUM(layer_i(x) * weight_i(x, y)) + bias_i(y)
            #                   ^ for every x in layer i

            # layer_input_batch is a 2D numpy array of (batch_size x layer_i size)
            # layer['weights'] is a 2D numpy array of (layer_i size x layer_i+1 size)

            # weight_sum is a 2D numpy array of (batch_size x layer_i+1 size)
            #   * For each input in the batch, we've calculated the sums that going into
            #     layer_i+1
            #   * Example: let j be the number of neurons in layer i, and k in layer i + 1
            #          [ sum(w(1, 1) -> w(j, 1))
            #            sum(w(1, 2) -> w(j, 2))
            #            sum(w(1, 3) -> w(j, 3))
            #            ...
            #            sum(w(1, k) -> w(j, k)) ]
            #     This is the value of one row in weight_sum
            weight_sum_batch = numpy.dot(layer_input_batch, layer['weights'])

            # Now, for each layer_i+1 sized array in weight_sum, add the biases to each
            # neuron
            layer_output_batch = weight_sum_batch + layer['biases']


            if i < len(self.layers) - 1:
                # Map R -> 0 to 1 via sigmoid function
                layer_output_batch = sigmoid(layer_output_batch)
            else:
                # ???
                layer_output_batch = softmax(layer_output_batch)

            # Current output is layer_i+1's input
            layer_input_batch = layer_output_batch

        final_predictions = layer_output_batch
        return final_predictions

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

    def accuracy(self, images, solutions):
        predictions = self.predict(images)

        predictions = self._convertOutputToResult(predictions)
        solutions = self._convertOutputToResult(solutions)

        accuracy = numpy.sum(predictions == solutions) / float(images.shape[0])
        return accuracy

    def back_propogate(self, images, solutions):

        # Use sqaured error ?

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

        # Diff is a an array of shape (batch_size x output layer size)
        diff = (predictions - solutions) / number_of_images
        gradients = []
        print("diff.shape", diff.shape)

        # Iterate backwards through the layers
        for i in range(len(self.layers) - 1, -1, -1):

            # layer_output is the previous layer's values
            if i != 0:
                layer_output = outputs[i - 1]
            else:
                layer_output = images

            # Get current weights (layer_i_size x layer_i+1_size )
            current_weights = self.layers[i]['weights']

            # We're going from layer i+1 to layer i,
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
