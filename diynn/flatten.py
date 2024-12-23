import numpy as np

class Flatten():

    def __init__(self):

        self.shape = None

    def __call__(self, input):

        return self.forward(input)
    
    def forward(self, input):
            self.shape = input.shape
            return np.reshape(input, (-1, input.shape[-1]))
    
    def backward(self, gradients):

        return np.reshape(gradients, (self.shape))