import pytest
import numpy as np
from diynn.activation import Sigmoid


def test_sigmoid_display():
    x = np.linspace(-10, 10, 100)
    sigmoid = Sigmoid()
    z = sigmoid.compute(x)

    assert z[0] == pytest.approx(0, abs=1e-4)
    assert z[-1] == pytest.approx(1, 1e-4)

    """
    import matplotlib.pyplot as plt
    plt.plot(x, z) 
    plt.xlabel("x") 
    plt.ylabel("Sigmoid(X)") 
    
    plt.show() 
    """


def test_sigmoid():
    x = np.random.uniform(size=(3, 3))

    sigmoid = Sigmoid()

    computed = sigmoid.compute(x)
    assert computed.shape == x.shape

    derivative = sigmoid.derivative(x)
    assert derivative.shape == x.shape
