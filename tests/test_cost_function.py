import pytest
from diynn.cost_function import CostFunction


@pytest.mark.parametrize('one_hot_labels', [(10, 4)], indirect=True)
@pytest.mark.parametrize('inputs', [(10, 4)], indirect=True)
def test_cost_function(one_hot_labels, inputs):
    """
    Compute total loss acrross 10 examples for four classes
    """

    cost = CostFunction(nclasses=4)
    total_loss = cost(inputs, one_hot_labels)
    assert isinstance(total_loss, float)
