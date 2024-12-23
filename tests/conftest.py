from pytest import fixture
import numpy as np
from typing import List, Tuple


@fixture(scope="function")
def inputs(request):
    params = [r for r in request.param]
    if isinstance(params[0], Tuple):
        params[0]  = list(params[0])
    if not isinstance(params[0], List):
        params[0] = [params[0]]
    return np.random.uniform(size=params[0] + [params[1]])


@fixture(scope="function")
def labels(request):
    params = [r for r in request.param]
    if isinstance(params[0], Tuple):
        params[0]  = list(params[0])
    if not isinstance(params[0], List):
        params[0] = [params[0]]
    return np.random.uniform(size=params[0] + [params[1]])


@fixture(scope="function")
def one_hot_labels(request):
    labels = np.random.choice(request.param[1], size=(request.param[0]))
    return np.eye(4)[labels]
