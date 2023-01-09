import pytest
import random

from callback import Callback


@pytest.fixture
def fn1():
    def fn(**kw):
        return random.random()

    return fn


@pytest.fixture
def fn_x2():
    def fn(i, **kw):
        return 2 * i

    return fn


def test_callback(fn1):
    cb = Callback(fn1)
    for i in range(10):
        cb(namespace=locals())


def test_callback_x2(fn_x2):
    cb = Callback(fn_x2)
    for i in range(3):
        cb(namespace=locals())
    expected = [0, 2, 4]
    for i in range(3):
        assert cb.values['fn'][i] == expected[i]

