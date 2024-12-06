import numpy as np


class MeanSquaredError:
    """
    Compute total loss across m example and n classes
    Matrices of shape (examples, classes)
    """

    def __init__(self):
        self.input = None
        self.labels = None

    def __call__(self, input: np.array, labels: np.array) -> float:
        return self.forward(input, labels)

    def forward(self, input: np.array, labels: np.array) -> float:
        """Forward pass of cost function.

        :param input: _description_
        :type input: np.array
        :param labels: _description_
        :type labels: np.array
        :return: _description_
        :rtype: float
        """

        assert input.shape == labels.shape
        self.input = input
        self.labels = labels
        difference = np.square(input - labels)
        return np.sum(difference) / input.shape[1]

    def backward(self):
        """Derivative of cost function with respect to the input.

        :return: _description_
        :rtype: _type_
        """

        return 2 * (self.input - self.labels) / self.input.shape[1]
