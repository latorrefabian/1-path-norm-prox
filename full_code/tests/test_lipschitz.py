import math
import pytest
import torch

from rsparse.lipschitz import Sequential, Linear, ELU, OneLayerNetwork


@pytest.fixture
def linear_2x3_a():
    """
    Linear module with weight matrix

    [2,  0]
    [0, -3]
    [0,  0]

    """
    linear = Linear(2, 3)
    linear.weight.data = torch.tensor([[2., 0.], [0., -3.], [0., 0.]])
    return linear


@pytest.fixture
def linear_3x2_a():
    """
    Linear module with weight matrix

    [2,  0, 0]
    [0, -3, 0]

    """
    linear = Linear(2, 3)
    linear.weight.data = torch.tensor([[2., 0., 0.], [0., -3., 0.]])
    return linear


@pytest.fixture
def linear_3x1_a():
    """
    Linear module with weight matrix

    [2, 0, 0]

    """
    linear = Linear(3, 1)
    linear.weight.data = torch.tensor([[2., 0., 0.]])
    return linear


@pytest.fixture
def seq_2x3x2_a(linear_2x3_a, linear_3x2_a):
    """
    Sequential module with linear layers and ELU activation.
    Dimension of layers (2, 3, 2).

    """
    return Sequential(
            linear_2x3_a, ELU(), linear_3x2_a)


@pytest.fixture
def seq_2x3x1_a(linear_2x3_a, linear_3x1_a):
    """
    Sequential module with linear layers and ELU activation.
    Dimension of layers (2, 3, 1).

    """
    return Sequential(
            linear_2x3_a, ELU(), linear_3x1_a)


@pytest.fixture
def one_layer_2x3x2_a(linear_2x3_a, linear_3x2_a):
    """
    One hidden layer, fully connected network with ELU activation,
    of class lipschitz.OneLayerNetwork. Dimension of layers (2, 3, 2).

    """
    network = OneLayerNetwork(2, 3, 2, ELU())
    network.W1.weight.data = linear_2x3_a.weight.data
    network.W2.weight.data = linear_3x2_a.weight.data
    return network


@pytest.fixture
def one_layer_2x3x2_b():
    """
    One hidden layer, fully connected network with ELU activation,
    of class lipschitz.OneLayerNetwork. Dimension of layers (2, 3, 2).

    """
    network = OneLayerNetwork(2, 3, 2, ELU())
    network.W1.weight.data = torch.tensor([
        [1., 2.],
        [2., 3.],
        [0., 1.],
    ])
    network.W2.weight.data = torch.tensor([
        [1., 2., -1.],
        [0., 1., 3]
    ])
    return network


@pytest.fixture
def one_layer_2x3x1_b():
    """
    One hidden layer, fully connected network with ELU activation,
    of class lipschitz.OneLayerNetwork. Dimension of layers (2, 3, 2).

    """
    network = OneLayerNetwork(2, 3, 2, ELU())
    network.W1.weight.data = torch.tensor([
        [1., 2.],
        [2., 3.],
        [0., 1.],
    ])
    network.W2.weight.data = torch.tensor([
        [1., 2., -1.],
    ])
    return network


@pytest.fixture
def one_layer_2x3x1_a(linear_2x3_a, linear_3x1_a):
    """
    One hidden layer, fully connected network with ELU activation,
    of class lipschitz.OneLayerNetwork. Dimension of layers (2, 3, 1).

    """
    network = OneLayerNetwork(2, 3, 1, ELU())
    network.W1.weight.data = linear_2x3_a.weight.data
    network.W2.weight.data = linear_3x1_a.weight.data
    return network


def test_constrain_2_a(seq_2x3x2_a):
    """
    Test if constraining the Lipschitz constant of a sequential module
    (layer-wise) to be less than one with respect to the L2-norm yields a
    module with Lipschitz constant at most one.

    """
    seq = seq_2x3x2_a
    seq.constrain(r=1., p=2)
    assert seq.lipschitz(p=2).item() <= 1.


def test_constrain_2_b(seq_2x3x1_a):
    """
    Test if constraining the Lipschitz constant of a sequential module
    (layer-wise) to be less than one with respect to the L2-norm yields a
    module with Lipschitz constant at most one.

    """
    seq = seq_2x3x1_a
    seq.constrain(r=1., p=2)
    assert seq.lipschitz(p=2).item() <= 1.


def test_constrain_inf_a(seq_2x3x2_a):
    """
    Test if constraining the Lipschitz constant of a sequential module
    (layer-wise) to be less than one with respect to the Linf-norm yields a
    module with Lipschitz constant at most one.

    """
    seq = seq_2x3x2_a
    seq.constrain(r=1., p=float('inf'))
    assert seq.lipschitz(p=float('inf')).item() <= 1.


def test_constrain_inf_b(seq_2x3x1_a):
    """
    Test if constraining the Lipschitz constant of a sequential module
    (layer-wise) to be less than one with respect to the Linf-norm yields a
    module with Lipschitz constant at most one.

    """
    seq = seq_2x3x1_a
    seq.constrain(r=1., p=float('inf'))
    assert seq.lipschitz(p=float('inf')).item() <= 1.


def test_lipschitz_linear(linear_2x3_a, linear_3x2_a, linear_3x1_a):
    """
    Test if the computation of the Lipschitz constant of linear layers
    is correct.

    """
    assert math.isclose(linear_2x3_a.lipschitz(p=2).item(), 3.)
    assert math.isclose(linear_2x3_a.lipschitz(p=float('inf')), 3.)

    assert math.isclose(linear_3x2_a.lipschitz(p=2).item(), 3.)
    assert math.isclose(linear_3x2_a.lipschitz(p=float('inf')), 3.)

    assert math.isclose(linear_3x1_a.lipschitz(p=2).item(), 2.)
    assert math.isclose(linear_3x1_a.lipschitz(p=float('inf')), 2.)


def test_lipschitz_sequential(seq_2x3x2_a, seq_2x3x1_a):
    """
    Test if the computation of the Lipschitz constant of a sequential
    module composed of linear layers, using the product upper bound,
    is correct.

    """
    assert math.isclose(seq_2x3x2_a.lipschitz(p=2).item(), 9.)
    assert math.isclose(seq_2x3x2_a.lipschitz(p=float('inf')), 9.)

    assert math.isclose(seq_2x3x1_a.lipschitz(p=2).item(), 6.)
    assert math.isclose(seq_2x3x1_a.lipschitz(p=float('inf')), 6.)


def test_one_layer_lipschitz(one_layer_2x3x1_a, one_layer_2x3x2_a):
    """
    Test if the computation of the Lipschitz constant of a one hidden layer
    network composed of linear layers, using the product upper bound, is
    correct.

    """
    lips_prod = one_layer_2x3x2_a.lipschitz(p=2, method='product').item()
    assert math.isclose(lips_prod, 9.)
    lips_prod = one_layer_2x3x1_a.lipschitz(p=2, method='product').item()
    assert math.isclose(lips_prod, 6.)


def test_path(
        one_layer_2x3x2_a, one_layer_2x3x2_b,
        one_layer_2x3x1_a, one_layer_2x3x1_b):
    """
    Test if the computation of the Lipschitz constant of a one hidden layer
    network composed of linear layers, using the path upper bound, is
    correct.

    """
    lips_path = one_layer_2x3x2_a.lipschitz(
            p=float('inf'), method='path').item()
    assert math.isclose(lips_path, 13.)
    lips_path = one_layer_2x3x2_b.lipschitz(
            p=float('inf'), method='path').item()
    assert math.isclose(lips_path, 22.)
    lips_path = one_layer_2x3x1_b.lipschitz(
            p=float('inf'), method='path').item()
    assert math.isclose(lips_path, 14.)

