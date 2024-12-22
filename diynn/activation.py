import math
import numpy as np


class Activation:
    def __init__(self): ...

    def compute(self, x: np.array) -> np.array:
        """
        Compute function

        :raises NotImplementedError: _description_
        """
        raise NotImplementedError

    def derivative(self, x: np.array) -> np.array:
        """
        Compute derivative

        :raises NotImplementedError: _description_
        """
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return self.compute(x)

    def compute(self, x: np.array) -> np.array:
        """Compute function on input

        :param x: input data
        :type x: np.array
        :return: function evaluated at x
        :rtype: np.array
        """

        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.array) -> np.array:
        """Compute function derivative on input

        :param x: input data
        :type x: np.array
        :return: function derivative evaluated at x
        :rtype: np.array
        """
        return self.compute(x) * (1 - self.compute(x))
