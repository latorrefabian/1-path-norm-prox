# import pdb
import pytest
import torch

from rsparse.lipschitz import FullyConnected, ELU
from rsparse.regularizer import L1


torch.manual_seed(1)


@pytest.fixture
def fc_w_bias():
    return FullyConnected(2, 3, 1, activation=ELU(), bias=True)


@pytest.fixture
def tensor1():
    return torch.tensor([[.4, .4], [.5, 1.]])


@pytest.fixture
def tensor2():
    return torch.tensor([[.4, .4], [.5, 1.], [.6, 1.]])


@pytest.fixture
def single_tensor1():
    return torch.tensor([.4, .4])


@pytest.fixture
def single_tensor2():
    return torch.tensor([.5, 1.])


def test_l1(fc_w_bias):
    l1 = L1(bias=True, requires_grad=False)
    val_w_bias = l1(fc_w_bias).item()
    l1.bias = False
    val_wo_bias = l1(fc_w_bias).item()
    assert val_wo_bias < val_w_bias


def test_l1_prox(fc_w_bias):
    l1 = L1(bias=True, requires_grad=False)
    l1.prox(lambda_=1.0, module=fc_w_bias)

