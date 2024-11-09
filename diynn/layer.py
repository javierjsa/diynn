import numpy as np


class NNLayer:
    """
    This class represent a linear layer
    """

    def __init__(self, ninputs: int, noutputs: int, activation: str):
        """
        Constructor

        :param ninputs: Dimension of input vector
        :type ninputs: int
        :param noutputs: Dimension of output vector
        :type noutputs: int
        :param activation: Numpy activation function
        :type activation: str
        """

        self.activation = getattr(np, activation)
        self.weights =  np.random.uniform(size=(noutputs, ninputs))
        self.biases = np.random.uniform(size=(noutputs,1))

    
    def forward(self, inputs: np.array) -> np.array:
        """
        Forward propagation

        :param inputs: input vector data
        :type inputs: np.array
        :return: layer output vector
        :rtype: np.array
        """

        return self.activation((self.weights @ inputs) + self.biases)
