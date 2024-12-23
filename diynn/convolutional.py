import numpy as np
import math as m

class Conv:


    def __init__(self, filters, size, channels, stride, padding):

        self.filters = filters
        self.size = size
        self.padding = 0 if padding == "valid" else m.ceil((self.size -1) /2)
        self.stride = stride
        self.output_size = None
        
        self.weights = np.random.uniform(0,1, size=(size,size, channels, filters))


    def __call__(self, inputs: np.ndarray):
        return self.forward(inputs)
    

    def forward(self, inputs: np.ndarray):

        input_size = inputs.shape[0]
        if not self.output_size:
            self.output_size = m.floor(1 + ((input_size - self.size + 2 * self. padding) / self.stride))
        index = 0
        output = np.zeros(shape=(self.output_size * self.output_size, self.filters, inputs.shape[-1]))
        for rows in range(0, inputs.shape[0], self.size):
            for cols in range(0, inputs.shape[1], self.size):                
                sub_input = inputs[rows: rows + self.size, cols: cols + self.size, :, :]
                print("------------------")
                print(f"{rows}: {rows + self.size}, {cols}: {cols + self.size}")
                print(index)
                for i in range(0, self.weights.shape[-1]):
                    corr = sub_input * self.weights[..., i][..., np.newaxis]
                    sum = np.sum(corr, axis=(0, 1, 2), keepdims=False)
                    output[index, i,:] = sum
                index += 1
            index +=1
        output = np.reshape(output, (self.output_size, self.output_size, self.filters, inputs.shape[-1]) )