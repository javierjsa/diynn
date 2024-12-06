import pytest
import numpy as np
from diynn.layer import NNLayer
from diynn.activation import Sigmoid
from diynn.cost_function import MeanSquaredError


@pytest.mark.parametrize("inputs", [(5, 1)], indirect=True)
def test_layer(inputs):
    """
    Test forward method
    """

    layer = NNLayer(ninputs=5, noutputs=10, activation=Sigmoid)
    output = layer.forward(inputs)

    assert output.shape == (10, 1)


@pytest.mark.parametrize("inputs", [(5, 1)], indirect=True)
def test_hidden_layer(inputs):
    """
    Test forward method with hidden layers
    """

    input_layer = NNLayer(ninputs=5, noutputs=10, activation=Sigmoid)
    hidden_layer = NNLayer(ninputs=10, noutputs=5, activation=Sigmoid)
    output_layer = NNLayer(ninputs=5, noutputs=2, activation=Sigmoid)

    layers = [input_layer, hidden_layer, output_layer]

    for layer in layers:
        inputs = layer.forward(inputs)

    assert inputs.shape == (2, 1)

    inputs = np.random.uniform(size=(5, 1))
    output = output_layer.forward(hidden_layer.forward(input_layer.forward(inputs)))

    assert output.shape == (2, 1)


@pytest.mark.parametrize("inputs", [(5, 1), (5, 10)], indirect=True)
def test_backward(inputs):
    """
    This test validates the architecture and dimensions of a 3-layer neural network.

    - Input: A single training example, represented as a column vector of shape (5, 1).
    - First layer:
    - Weights: Shape (10, 5), where 10 is the number of neurons, and 5 is the number of input features.
    - Bias: Shape (10, 1), corresponding to the 10 neurons.
    - Hidden layer:
    - Weights: Shape (5, 10), where 5 is the number of neurons in the hidden layer, and 10 is the number of inputs from the previous layer.
    - Bias (if used): Shape (5, 1).
    - Output layer:
    - Weights: Shape (2, 5), where 2 is the number of neurons in the output layer, and 5 is the number of inputs from the hidden layer.
    - Bias (if used): Shape (2, 1).
    - Network Output: Shape (2, 1), where 2 corresponds to the output neurons and 1 corresponds to the single training example.
    """

    input_layer = NNLayer(ninputs=5, noutputs=10, activation=Sigmoid)
    hidden_layer = NNLayer(ninputs=10, noutputs=5, activation=Sigmoid)
    output_layer = NNLayer(ninputs=5, noutputs=2, activation=Sigmoid)

    layers = [input_layer, hidden_layer, output_layer]

    for layer in layers:
        inputs = layer.forward(inputs)

    gradient = layers[2].backward()
    assert gradient.shape == layers[2].inputs.shape

    gradient = layers[1].backward(gradient)
    assert gradient.shape == layers[1].inputs.shape

    gradient = layers[0].backward(gradient)
    assert gradient.shape == layers[0].inputs.shape


@pytest.mark.parametrize(
    "inputs, labels", [((5, 1), (2, 1)), ((5, 10), (2, 10))], indirect=True
)
def test_backward_with_cost(inputs, labels):
    input_layer = NNLayer(ninputs=5, noutputs=10, activation=Sigmoid)
    hidden_layer = NNLayer(ninputs=10, noutputs=5, activation=Sigmoid)
    output_layer = NNLayer(ninputs=5, noutputs=2, activation=Sigmoid)
    cost = MeanSquaredError()

    layers = [input_layer, hidden_layer, output_layer]

    for layer in layers:
        inputs = layer(inputs)

    loss = cost(inputs, labels)

    assert loss >= 0

    gradient = cost.backward()

    gradient = layers[2].backward(gradient)
    assert gradient.shape == layers[2].inputs.shape

    gradient = layers[1].backward(gradient)
    assert gradient.shape == layers[1].inputs.shape

    gradient = layers[0].backward(gradient)
    assert gradient.shape == layers[0].inputs.shape
