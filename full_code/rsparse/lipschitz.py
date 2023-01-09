# import pdb
import torch

from abc import ABC, abstractmethod
from torch import nn

from . import projection


class Lipschitz(ABC):
    """
    Abstract base class for a Lipschitz continuous set of functions.
    Should provide the functionality of constraining the Lipschitz constant,
    as well as computing it, preferrably in such a way that it is compatible
    with automatic differentiation modules.

    """
    @abstractmethod
    def constrain(self, r, p):
        """
        Constrain the Lipschitz constant to be at most r

        Args:
            r (float): constrain on the Lipschitz constant
            p (int or float('inf')): L_p-norm to use

        """
        pass

    @abstractmethod
    def lipschitz(self, p, **kwargs):
        """
        Compute an upper bound on the Lipschitz constant of the network

        Args:
            p (int or float('inf')): L_p-norm to use
            **kwargs: extra parameters that determine exactly how the
                upper bound on the Lipschitz constant is computed

        Returns:
            (torch.tensor): upper bound on the Lipschitz constant

        """
        pass


class Linear(nn.Linear, Lipschitz):
    """
    Linear module with handling of Lipschitz constant
    """
    def constrain(self, r, p):
        if p == 2:
            self.weight.data = projection.proj_2_matrix_norm(
                    self.weight.data, r=r)
        elif p == float('inf'):
            self.weight.data = projection.proj_inf_matrix_norm(
                    self.weight.data, r=r)
        else:
            raise NotImplementedError(
                    'p-norm Lipschitz constraint '
                    'not implemented for p=' + str(p))

    def lipschitz(self, p, **kwargs):
        if p == 2:
            u, s, v = torch.svd(self.weight, some=True)
            return s[0]
        elif p == float('inf'):
            return torch.max(torch.norm(self.weight, p=1, dim=1))


class Sequential(nn.Sequential, Lipschitz):
    """
    Sequential module with handling of Lipschitz constant
    """
    def constrain(self, r, p):
        for module in self._modules.values():
            module.constrain(r=r, p=p)

    def lipschitz(self, p, method='product', **kwargs):
        """
        Args:
            method (str): specifies which upper bound of the Lipschitz constant
                will be computed. Currently supported: ('product') upper bounds
                the constant by the product of the constants of each component

        """
        if method == 'product':
            lips = 1.
            for module in self._modules.values():
                lips *= module.lipschitz(p=p)
            return lips
        else:
            raise ValueError('Method ' + str(method) + ' not recognized')

    def layerwise_lipschitz(self, p):
        """
        Compute an upper bound on the Lipschitz constant per component of
        the sequential module

        Args:
            p (int or float): defines the Lp norm used for the computation of
                the constants

        Returns:
            (list): upper bound on the constant for each component

        """
        return [m.lipschitz(p=p) for m in self._modules.values()]


class OneLayerNetwork(nn.Module, Lipschitz):
    """
    Fully connected network with a single hidden layer

    Args:
        in_features (int): dimension of input space
        hidden_features (int): size of hidden layer
        out_features (int): dimension of output space
        activation (nn.Module): activation function module
        bias_1 (bool): whether to use bias in the hidden layer (default: True)
        bias_2 (bool): whether to use bias in the second layer (default: True)

    """
    def __init__(
            self, in_features, hidden_features, out_features, activation,
            bias_1=True, bias_2=True):
        super().__init__()
        self.W1 = Linear(in_features, hidden_features, bias=bias_1)
        self.W2 = Linear(hidden_features, out_features, bias=bias_2)
        self.activation = activation
        self.fc = Sequential(self.W1, activation, self.W2)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))

    def constrain(self, r, p):
        self.fc.constrain(r, p)

    def lipschitz(self, p, method='product', **kwargs):
        """
        Args:
            method (str): specifies which upper bound of the Lipschitz constant
                will be computed. Currently supported: ('path') upper bounds
                the constant by the sum of the product of weights (absolute
                value) along any path from input to output. Any other method
                implemented by the class 'Sequential' is also available.

        """
        if method == 'path':
            if p == float('inf'):
                paths = self.W1.weight * self.W2.weight[:, :, None]
                return torch.abs(paths).sum()
            else:
                raise ValueError(
                        "'path' Lipschitz constant method only implemented"
                        " for p=float('inf')")
        else:
            return self.fc.lipschitz(p=p, method=method, **kwargs)

    def layerwise_lipschitz(self, p):
        """
        See Sequential.layerwise_lipschitz method
        """
        return self.fc.layerwise_lipschitz(p=p)


class ThreeLayerNetwork(nn.Module, Lipschitz):
    """
    Fully connected network with a three hidden layers

    Args:
        in_features (int): dimension of input space
        hidden_features (int*): list with sizes of hidden layers
        out_features (int): dimension of output space
        activation (nn.Module): activation function module
        bias_1 (bool): whether to use bias in the first layer (default: True)
        bias_2 (bool): whether to use bias in the second layer (default: True)
        bias_3 (bool): whether to use bias in the third layer (default: True)
        bias_4 (bool): whether to use bias in the output layer (default: True)

    """
    def __init__(
            self, in_features, hidden_features, out_features, activation,
            bias_1=True, bias_2=True, bias_3=True, bias_4=True):
        super().__init__()
        self.W1 = Linear(in_features, hidden_features[0], bias=bias_1)
        self.W2 = Linear(hidden_features[0], hidden_features[1], bias=bias_2)
        self.W3 = Linear(hidden_features[1], hidden_features[2], bias=bias_3)
        self.W4 = Linear(hidden_features[2], out_features, bias=bias_4)
        self.activation = activation
        self.fc = Sequential(
                self.W1, activation,
                self.W2, activation,
                self.W3, activation,
                self.W4)

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))

    def constrain(self, r, p):
        self.fc.constrain(r, p)

    def lipschitz(self, p, method='product', **kwargs):
        """
        Args:
            method (str): specifies which upper bound of the Lipschitz constant
                will be computed. Currently supported: ('path') upper bounds
                the constant by the sum of the product of weights (absolute
                value) along any path from input to output splitting by
                pairwise consecutive layers. Any other method implemented by
                the class 'Sequential' is also available.

        """
        if method == 'path':
            if p == float('inf'):
                paths1 = self.W1.weight * self.W2.weight[:, :, None]
                paths2 = self.W3.weight * self.W4.weight[:, :, None]
                return torch.abs(paths1).sum() + torch.abs(paths2).sum()
            else:
                raise ValueError(
                        "'path' Lipschitz constant method only implemented"
                        " for p=float('inf')")
        else:
            return self.fc.lipschitz(p=p, method=method, **kwargs)

    def layerwise_lipschitz(self, p):
        """
        See Sequential.layerwise_lipschitz method
        """
        return self.fc.layerwise_lipschitz(p=p)


class FullyConnected(nn.Module, Lipschitz):
    def __init__(self, *dimensions, bias, activation):
        """
        Fully Connected neural network

        Args:
            *dimensions (int): dimensions of layers, from input to output
            bias (bool): If True, each layer will have a bias term
            activation (nn.Module): activation function

        """
        super().__init__()
        self.layers = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            self.layers.append(Linear(input_dim, output_dim, bias=bias))
            self.layers.append(activation)
        self.fc = Sequential(*self.layers[:-1])

    def forward(self, x):
        return self.fc(x.view(x.shape[0], -1))

    def constrain(self, r, p):
        self.fc.constrain(r, p)

    def lipschitz(self, p, method='product', **kwargs):
        return self.fc.lipschitz(p, method=method)


class Conv2d(nn.Conv2d, Lipschitz):
    """
    2d convolutional layer with handling of Lipschitz constant
    """
    def constrain(self, r, p):
        raise NotImplementedError

    def lipschitz(self, p, **kwargs):
        raise NotImplementedError


class ReLU(nn.ReLU, Lipschitz):
    """
    ReLU activation with handling of Lipschitz constant
    """
    def constrain(self, r, p):
        """
        ReLU has no trainable parameters
        """
        pass

    def lipschitz(self, p, **kwargs):
        return 1.


class ELU(nn.ELU, Lipschitz):
    """
    ELU activation with handling of Lipschitz constant
    """
    def constrain(self, r, p):
        """
        ELU has no trainable parameters
        """
        pass

    def lipschitz(self, p, **kwargs):
        return self.alpha

