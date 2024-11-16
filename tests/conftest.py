from pytest import fixture
import numpy as np



@fixture(scope="function")
def inputs(request):

    return np.random.uniform(size=(request.param[0],request.param[1]))

@fixture(scope="function")
def labels(request):

    return np.random.choice(request.param[1], size=(request.param[0], 1))


@fixture(scope="function")
def one_hot_labels(request):

    labels = np.random.choice(request.param[1], size=(request.param[0]))
    return np.eye(4)[labels]
