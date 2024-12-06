import numpy as np
from diynn.activation import Activation


class NNLayer:
    """
    This class represent a linear layer
    """

    def __init__(self, ninputs: int, noutputs: int, activation: Activation):
        """
        Constructor

        :param ninputs: Dimension of input vector
        :type ninputs: int
        :param noutputs: Dimension of output vector
        :type noutputs: int
        :param activation: Numpy activation function
        :type activation: str
        """

        self.activation = activation()
        self.weights = np.random.uniform(size=(noutputs, ninputs))
        self.biases = np.random.uniform(size=(noutputs, 1))
        self.inputs = None
        self.outputs = None
        self.weights_grad = None
        self.bias_grad = None

    def __call__(self, inputs: np.array) -> np.array:
        return self.forward(inputs)

    def forward(self, inputs: np.array) -> np.array:
        """
        Forward propagation

        :param inputs: input vector data
        :type inputs: np.array
        :return: layer output vector
        :rtype: np.array
        """

        self.inputs = inputs
        self.outputs = self.activation.compute((self.weights @ inputs) + self.biases)
        return self.outputs

    def backward(self, gradient=None):
        """
        Performs the backward pass, computing the gradients for weights and biases.

        Arguments:
        - gradient: The gradient propagated from the subsequent layer, with shape matching the output of the layer.

        Steps:
        1. Compute the derivative of the activation function (same shape as the output).
        2. Perform element-wise multiplication (Hadamard product) between the propagated gradient and the activation derivative to get the combined gradient.
        - The Hadamard product is used to scale the gradient for each neuron based on the local activation derivative.
        3. Compute the gradient for the weights:
        - Multiply the combined gradient (shape: (n_neurons, 1)) with the transpose of the inputs (shape: (1, n_inputs)) using the regular matrix product (`@`).
        - The matrix product distributes the neuron-wise gradient across all input dimensions, resulting in a weight gradient with the same shape as the weights: (n_neurons, n_inputs).
        4. Compute the gradient for the biases:
        - The combined gradient directly represents the bias gradient, with the same shape as the biases: (n_neurons, 1).
        """

        pass_on_gradient = self.weights.T @ self.activation.derivative(self.outputs)
        if gradient is not None:
            self.weights_grad = (
                gradient * self.activation.derivative(self.outputs) @ self.inputs.T
            )
            self.bias_grad = gradient * self.activation.derivative(self.outputs)

        return pass_on_gradient
