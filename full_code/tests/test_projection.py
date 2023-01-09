import pytest
import torch

from rsparse.projection import proj_l1, soft_thresh


@pytest.fixture
def matrix_2x2_a():
    return torch.tensor([[.4, .4], [.5, 1.]])


@pytest.fixture
def matrix_2x2_b():
    return torch.tensor([[.4, -.4], [-.5, 1.]])


@pytest.fixture
def matrix_3x2_a():
    return torch.tensor([[.4, .4], [.5, 1.], [.6, 1.]])


@pytest.fixture
def matrix_3x2_b():
    return torch.tensor([[.4, -.4], [-.5, 1.], [.6, -1.]])


@pytest.fixture
def matrix_2x3_a():
    return torch.tensor([[2.,  0.,  0.], [0., -3.,  0.]])


@pytest.fixture
def matrix_1x3_a():
    return torch.tensor([[0., -2.,  0.]])


def test_l1_proj_1x3_a(matrix_1x3_a):
    result = proj_l1(matrix_1x3_a, r=1.)
    expected = torch.tensor([[0., -1., 0.]])
    assert torch.allclose(result, expected)


def test_l1_proj_2x2_a(matrix_2x2_a):
    result = proj_l1(matrix_2x2_a, r=1.)
    expected = torch.tensor([[.4, .4], [.25, .75]])
    assert torch.allclose(result, expected)


def test_l1_proj_2x2_b(matrix_2x2_b):
    result = proj_l1(matrix_2x2_b, r=1.)
    expected = torch.tensor([[.4, -.4], [-.25, .75]])
    assert torch.allclose(result, expected)


def test_l1_proj_2x3_a(matrix_2x3_a):
    result = proj_l1(matrix_2x3_a, r=1.)
    expected = torch.tensor([[1., 0., 0.], [0., -1., 0.]])
    assert torch.allclose(result, expected)


def test_l1_proj_3x2_a(matrix_3x2_a):
    result = proj_l1(matrix_3x2_a, r=1.)
    expected = torch.tensor([[.4, .4], [.25, .75], [.3, .7]])
    assert torch.allclose(result, expected)


def test_l1_proj_3x2_b(matrix_3x2_b):
    result = proj_l1(matrix_3x2_b, r=1.)
    expected = torch.tensor([[.4, -.4], [-.25, .75], [.3, -.7]])
    assert torch.allclose(result, expected)


def test_soft_thresh(matrix_2x2_a):
    lambda_ = torch.tensor([.5, .6])
    result = soft_thresh(lambda_, matrix_2x2_a)
    expected = torch.tensor([[0., 0.], [0., .4]])
    assert torch.allclose(result, expected)
    lambda_ = 0.1
    result = soft_thresh(lambda_, matrix_2x2_a)
    expected = torch.tensor([[.3, .3], [.4, .9]])
    assert torch.allclose(result, expected)

