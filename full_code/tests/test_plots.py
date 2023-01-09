import numpy as np
import os
import pytest

from plots import lineplot


@pytest.fixture
def data_x_y_1():
    data = {
            'A': ([1., 2., 3., 4., 5.], [1., 2., 3., 4., 5.]),
            'B': ([1., 1., 1., 2., 3.], [1., 2., 3., 3., 3.]),
            'C': ([0., 1., 2., 3., 4.], [0., 1., 2., 3., 4.])
    }

    for k, v in data.items():
        data[k] = [np.array(x) for x in v]

    return data


@pytest.fixture
def data_x_y_2():
    data = {
            'A': ([5., 4., 3., 2., 1.], [1., 2., 3., 4., 5.]),
            'B': ([1., 1., 1., 2., 3.], [1., 2., 3., 3., 3.]),
            'C': ([4., 3., 2., 1., 0.], [0., 1., 2., 3., 4.])
    }

    for k, v in data.items():
        data[k] = [np.array(x) for x in v]

    return data


@pytest.fixture
def data_y_1():
    data = {
            'A': ([1., 2., 3., 4., 5.],),
            'B': ([1., 2., 3., 3., 3.],),
            'C': ([0., 1., 2., 3., 4.],)
    }

    for k, v in data.items():
        data[k] = [np.array(x) for x in v]

    return data


@pytest.fixture
def data_y_2():
    data = {
            'A': ([5., 4., 3., 2., 1.],),
            'B': ([1., 1., 1., 2., 3.],),
            'C': ([4., 3., 2., 1., 0.],)
    }

    for k, v in data.items():
        data[k] = [np.array(x) for x in v]

    return data


@pytest.fixture
def multi_data_x_y(data_x_y_1, data_x_y_2):
    return {
        'z=1': data_x_y_1,
        'z=2': data_x_y_2,
    }


@pytest.fixture
def multi_data_y(data_y_1, data_y_2):
    return {
        'z=1': data_y_1,
        'z=2': data_y_2,
    }


def test_plot_single_x_y(data_x_y_1):
    f = 'test_plot_single_x_y.pdf'
    lineplot(
            data_x_y_1, xlabel='x', ylabel='y', xscale='linear',
            yscale='linear',
            filename='test_plot_single_x_y.pdf', conference='icml',
            size='one_column')
    assert os.path.isfile(f)
    os.remove(f)


def test_plot_single_y(data_y_1):
    f = 'test_plot_single_y.pdf'
    lineplot(
            data_y_1, xlabel='iteration', ylabel='y', xscale='linear',
            yscale='linear',
            filename='test_plot_single_y.pdf', conference='icml',
            size='one_column')
    assert os.path.isfile(f)
    os.remove(f)


def test_plot_multi_x_y(multi_data_x_y):
    f = 'test_plot_multi_x_y.pdf'
    lineplot(
            multi_data_x_y, xlabel='x', ylabel='y', xscale='linear',
            yscale='linear',
            filename='test_plot_multi_x_y.pdf', conference='icml',
            size='one_column')
    assert os.path.isfile(f)
    os.remove(f)


def test_plot_multi_y(multi_data_y):
    f = 'test_plot_multi_y.pdf'
    lineplot(
            multi_data_y, xlabel='iteration', ylabel='y', xscale='linear',
            yscale='linear',
            filename=f, conference='icml',
            size='one_column')
    assert os.path.isfile(f)
    os.remove(f)

