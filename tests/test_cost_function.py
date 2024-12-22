import pytest
from diynn.cost_function import MeanSquaredError


@pytest.mark.parametrize("labels", [(10, 4)], indirect=True)
@pytest.mark.parametrize("inputs", [(10, 4)], indirect=True)
def test_cost_function(labels, inputs):
    """
    Compute total loss acrross 10 examples for four classes
    """

    cost = MeanSquaredError()
    total_loss = cost(inputs, labels)
    assert isinstance(total_loss, float)


@pytest.mark.parametrize("labels", [(10, 4)], indirect=True)
@pytest.mark.parametrize("inputs", [(10, 4)], indirect=True)
def test_cost_function_backward(labels, inputs):
    cost = MeanSquaredError()
    total_loss = cost(inputs, labels)
    assert total_loss >= 0
    gradient = cost.backward()

    assert gradient.shape == inputs.shape
