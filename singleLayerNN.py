from numpy import exp, array, dot, random


class NeuralNetwork():
    def __init__(self):
        # Seed random number generator, as it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values from -1 to 1
        # and a mean of 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # The sigmoid function, which describes an s shaped curve.
    # We pass the weighted sum of the inputs through the function
    # to normalize them between 0 and 1.
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

        # Gradient of the sigmoid curve
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs):
        # pass inputs through our sneural network (single neuron)
        return self.sigmoid(dot(inputs, self.synaptic_weights))

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            output = self.think(training_set_inputs)

            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the
            # sigmoid curve
            adjustment = dot(training_set_inputs.T, error + self.sigmoid_derivative(output))

            # Adjust the weights
            self.synaptic_weights += adjustment


if __name__ == '__main__':
    neural_network = NeuralNetwork()

    print 'Random starting synaptic weights:'
    print neural_network.synaptic_weights

    # Training set
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print 'New synaptic weights after training:'
    print neural_network.synaptic_weights

    # Test new condition
    print 'Consider new situation [1, 0 , 0] -> ?:'
    print neural_network.think(array([1, 0, 2]))
