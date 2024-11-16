import numpy as np

class CostFunction():
    """
    Compute total loss across m example and n classes
    Matrices of shape (examples, classes)
    """

    def __init__(self, nclasses):
        
        self.nclasses = nclasses

    def __call__(self, input: np.array, labels: np.array) -> float:
        
        difference = np.square(input - labels)
        return np.sum(difference) / input.shape[0]