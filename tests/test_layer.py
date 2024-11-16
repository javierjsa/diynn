import pytest
import numpy as np
from diynn.layer import NNLayer


@pytest.mark.parametrize('inputs', [(5,1)], indirect=True)
def test_layer(inputs):
    """
    Test forward method
    """

    layer = NNLayer(ninputs=5, noutputs=10, activation="tanh")
    output = layer.forward(inputs)
    
    assert output.shape == (10,1)


@pytest.mark.parametrize('inputs', [(5,1)], indirect=True)
def test_hidden_layer(inputs):
    """
    Test forward method with hidden layers
    """

    input_layer = NNLayer(ninputs=5, noutputs=10, activation="tanh")
    hidden_layer = NNLayer(ninputs=10, noutputs=5, activation="tanh")
    output_layer = NNLayer(ninputs=5, noutputs=2, activation="tanh")

    layers = [input_layer, hidden_layer, output_layer]

    for layer in layers:
        inputs = layer.forward(inputs)

    assert inputs.shape == (2,1)

    inputs = np.random.uniform(size=(5,1))
    output = output_layer.forward(hidden_layer.forward(input_layer.forward(inputs)))
        
    assert output.shape == (2,1)