import pytest
from diynn.convolutional import Conv



def test_convolutional():

    conv = Conv(filters=3, size=5, channels=3, stride=3, padding='zeros')
    assert conv.weights.shape == (5,5,3,3)

@pytest.mark.parametrize("inputs", [((25, 25, 3), 10)], indirect=True)
def test_convolutional_forward(inputs):

    conv = Conv(filters=3, size=5, channels=3, stride=5, padding='valid')
    output =  conv(inputs)
    assert output.shape == (5,5,3)

