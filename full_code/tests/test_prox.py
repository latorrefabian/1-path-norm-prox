import pdb
import copy
import math
import pytest
import torch

from rsparse.lipschitz import OneLayerNetwork, ELU
from rsparse.regularizer import PathLength2
from rsparse.proximal import prox_positive_factory, _k_vector, _k_matrix

from rsparse.utils import h_path_mult

torch.manual_seed(1)


@pytest.fixture
def x_test():
    return torch.tensor([[1., 2., 3.]])


@pytest.fixture
def Y_test():
    return torch.tensor(
        [[1., 2., 3., 4],
         [3., 4., 5., 6.],
         [5., 6, 7., 8.]]
    )


@pytest.fixture
def lambda_():
    return 1.


@pytest.fixture
def network():
    return OneLayerNetwork(4, 3, 2, activation=ELU())


def test_path_prox(network):
    lambda_ = 0.1
    path = PathLength2()
    vbar = network.W2.weight.data
    wbar = network.W1.weight.data
    path.prox(lambda_=lambda_, module=network)

    for i in range(10):
        V = torch.rand(vbar.shape)
        W = torch.rand(wbar.shape)
        h = vbar.shape[1]  # number of hidden neurone
        for j in range(h):
            f = h_path_mult(vbar[:, j], wbar[j, :], lambda_)
            v = network.W2.weight.data[:, j]
            w = network.W1.weight.data[j, :]
            v_rnd = V[:, j]
            w_rnd = W[j, :]
            assert f(v, w) < f(v_rnd, w_rnd)

    network.W1.weight.data = torch.tensor(
            [[0.4116,  0.1772, -0.2248, 0.1030],
             [0.3919, -0.0149,  0.1139,  0.4613],
             [-0.2737,  0.3785, -0.0665, -0.0034]])
    network.W2.weight.data = torch.tensor(
            [[0.4999, 0.5504, -0.2444],
             [-0.0930,  0.5427, -0.2093]])
    w_target = torch.tensor(
            [[0.3671,  0.1327, -0.1802,  0.0585],
             [0.2962, -0.0000,  0.0182,  0.3656],
             [-0.2407,  0.3455, -0.0335, -0.0000]])
    v_target = torch.tensor(
            [[0.4261,  0.4824, -0.1825],
             [-0.0191,  0.4747, -0.1473]])
    path.prox(lambda_=lambda_, module=network)
    assert (network.W1.weight.data - w_target).norm() < 1e-3
    assert (network.W2.weight.data - v_target).norm() < 1e-3


def test_path_prox_fast(network):
    lambda_ = 0.1
    path = PathLength2()
    network_copy = copy.deepcopy(network)

    path.prox(lambda_=lambda_, module=network, fast=False)
    v_slow = network.W2.weight.data
    w_slow = network.W1.weight.data

    path.prox(lambda_=lambda_, module=network_copy, fast=True)
    v_fast = network.W2.weight.data
    w_fast = network.W1.weight.data

    assert(torch.sum(v_slow - v_fast) < 1e-10)
    assert(torch.sum(w_slow - w_fast) < 1e-10)


def test_prox_factory(x_test, Y_test, lambda_):
    prox_positive_factory(
            m=Y_test.shape[0], lambda_=lambda_, device=x_test.device)


def test_k_vector():
    s_max = 3
    lambda_ = 10.
    tol = 1e-7
    result = _k_vector(s_max, lambda_, device=torch.device('cpu'))
    assert len(result) == 4
    assert math.isclose(result[0].item(), 1., rel_tol=tol)
    assert math.isclose(result[1].item(), 1. + 1. / (1 - 100), rel_tol=tol)
    assert math.isclose(result[2].item(), 1. + 1. / (1 - 200), rel_tol=tol)
    assert math.isclose(result[3].item(), 1. + 1. / (1 - 300), rel_tol=tol)


def test_k_matrix():
    k = torch.arange(1., 5.)
    result = _k_matrix(k)
    expected = torch.tensor([
        [1., 0., 0, 0],
        [2, -1, 0, 0],
        [6, -3, -1, 0],
        [24, -12, -4, -1]
    ])
    assert torch.norm(result - expected).item() < 1e-3

