import pytest

from rsparse.utils import (
        params_to_filename, filename_to_params, SEPARATOR)


@pytest.fixture
def params_a():
    return dict(lr=0.01, epochs=10, cuda=False)


@pytest.fixture
def filename_a():
    f = 'bs' + SEPARATOR + '10' + SEPARATOR
    f += 'prox' + SEPARATOR + 'True'
    return f


def test_params_to_filename_to_params(params_a):
    """
    Test that transforms some dictionary of parameters to a filename and then
    back, in this case we should recover the exact same parameter names and
    values.

    """
    filename = params_to_filename(**params_a)
    params = filename_to_params(filename)

    for k, v in params_a.items():
        assert params_a[k] == params[k]

    for k, v in params.items():
        assert params_a[k] == params[k]


def test_filename_to_params_to_filename(filename_a):
    """
    Test that transforms some filename to a dictionary of paramters and then
    back, in this case we should recover the exact same filename.

    """
    params = filename_to_params(filename_a)
    filename = params_to_filename(**params)
    assert filename == filename_a

