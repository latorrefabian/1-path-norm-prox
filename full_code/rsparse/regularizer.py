# import pdb
from abc import ABC, abstractmethod
from rsparse.projection import soft_thresh

import torch
import numpy as np

from rsparse.utils import h_path_mult
from rsparse.utils import binary_search_2D


class Proximable(ABC):
    """
    Function with access to an efficient proximal operator oracle
    """
    @abstractmethod
    def prox(self, lambda_, module):
        pass


class Regularizer:
    """
    Regularization for neural networks built with pytorch modules

    Args:
        requires_grad (bool): if True it will add the regularizer to the
            computation graph. Should be False when using proximal gradient
            descent optimization algorithms.

    """
    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad

    def _call(self, module):
        raise NotImplementedError

    def __call__(self, module):
        """Evaluate the regularizer on a module

        Args:
            module (torch.nn.Module): module to regularize

        Returns:
            (torch.tensor): output of the regularizer on a given module

        """
        if self.requires_grad:
            return self._call(module)
        else:
            with torch.no_grad():
                return self._call(module).detach()


class L1(Regularizer, Proximable):
    """
    Proximable regularizer corresponding to the L1 norm of the parameters
    """
    def __init__(self, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.bias = bias

    def _call(self, module):
        """
        Sums the L1 norm of each parameter in the network. If self.bias is
        False, then bias parameters are ignored.

        Args:
            module (torch.nn.Module): module for which the L1 norm of its
                parameters will be computed

        Returns:
            (torch.tensor): value of total L1 norm of parameters

        """
        x = 0.
        for name, param in module.named_parameters():
            if 'bias' in name and not self.bias:
                continue
            x += torch.norm(param, p=1)

        return x

    def prox(self, lambda_, module):
        """
        Compute the proximal (soft-thresholding) and modify the parameters in
        place

        Args:
            lambda_ (float): multiplier of the proximal map
        """
        for name, param in module.named_parameters():
            if 'bias' in name and not self.bias:
                continue
            param.data = soft_thresh(lambda_, param.data)


class PathLength2(Regularizer, Proximable):
    """
    Proximable regularizer corresponding to the L1 path norm of a network
    grouped in pairwise consecutive layers
    """
    def _call(self, module):
        weights = []
        names = []
        for n, w in module.named_parameters():
            if 'weight' in n:
                weights.append(w)
                names.append(n)
        sum = 0
        for W1, W2 in zip(weights[0::2], weights[1::2]):
            l1norm = torch.norm(weights[0], p=1, dim=1)
            prod = torch.abs(weights[1]) * l1norm
            sum += prod.sum()
        return sum
#         return torch.abs(weights[1][:, :, None] * weights[0]).sum()

    def prox(
            self, lambda_, module, sparsity_diff_v=-1,
            sparsity_diff_w=-1, fast=True):
        """
        Compute the proximal (soft-thresholding) and modify the parameters in
        place.

        Args:
            lambda_ (float): multiplier of the proximal map
            module (rsparse.lipschitz.OneLayerNetwork): neural network
            sparsity_diff_v (int): the prox computation will check solutions
                for v that have s_v +- sparsity_diff_v nonzero entries, where
                s_v is the number of nonzero entries of the argument 'v' of the
                prox.
            sparsity_diff_w (int): the prox computation will check solutions
                for v that have s_w +- sparsity_diff_w nonzero entries, where
                s_w is the number of nonzero entries of the argument 'w' of the
                prox.
            fast (bool): fast version (TODO: explain a bit more)

            range of non-zero entries with respect to the  checked during prox
            computations

        """
        if "FullyConnected" in str(type(module)):
            linear_layers = module.layers[::2]  # remove activation layers
            for W1, W2 in zip(linear_layers[0::2], linear_layers[1::2]):
                W2.weight.data, W1.weight.data = self.prox_full(
                    W2.weight.data, W1.weight.data, lambda_)
        elif "OneLayerNetwork" in str(type(module)):
            module.W2.weight.data, module.W1.weight.data = self.prox_full(
                    module.W2.weight.data, module.W1.weight.data, lambda_)
        elif "ThreeLayerNetwork" in str(type(module)):
            module.W2.weight.data, module.W1.weight.data = self.prox_full(
                    module.W2.weight.data, module.W1.weight.data, lambda_)
            module.W4.weight.data, module.W3.weight.data = self.prox_full(
                    module.W4.weight.data, module.W3.weight.data, lambda_)
        else:
            raise NotImplementedError

    def prox_full(
            self, vbar, wbar, l, s_v_min=1, s_v_max=-1, s_w_min=1,
            s_w_max=-1, fast=True):
        v = torch.zeros(vbar.shape)
        w = torch.zeros(wbar.shape)
        h = vbar.shape[1]
        for i in range(h):
            # For loop over hidden neurons
            if fast:
                v_, w_ = self.prox_one_neuron_fast(
                    np.array(vbar[:, i]), np.array(wbar[i, :]), l)
            else:
                v_, w_ = self.prox_one_neuron(
                        np.array(vbar[:, i]), np.array(wbar[i, :]),
                        l, s_v_min[i], s_v_max[i], s_w_min[i], s_w_max[i])
            v[:, i] = torch.tensor(v_)
            w[i, :] = torch.tensor(w_)
        return v, w

    def prox_one_neuron_fast(self, vbar, wbar, l):
        """
        Prox for one hidden neuron

        Args:
            vbar: argument of prox
            wbar: argument of prox
            l: lambda
            s_v_min: minimum sparsity for v
            s_v_max: maximum sparsity for v
            s_w_min: minimum sparsity for w
            s_w_max: maximum sparsity for w
        """
        signs_v = np.sign(vbar)
        signs_w = np.sign(wbar)
        vbar = np.abs(vbar)
        wbar = np.abs(wbar)
        sorting_permutation_v = np.argsort(vbar)
        vbar = vbar[sorting_permutation_v]
        sorting_permutation_w = np.argsort(wbar)
        wbar = wbar[sorting_permutation_w]

        cumsum_vbar = np.cumsum(vbar[::-1])
        cumsum_wbar = np.cumsum(wbar[::-1])

        # Find possible optimal sparsities
        def h_vs(sv, sw):
            result = (1 - sv * sw * l ** 2) * vbar[-sv]
            result += l ** 2 * sw * cumsum_vbar[sv-1]
            result += - l * cumsum_wbar[sw-1]
            return result

        def h_ws(sv, sw):
            result = (1 - sv * sw * l ** 2) * wbar[-sw]
            result += l ** 2 * sv * cumsum_wbar[sw-1]
            result += - l * cumsum_vbar[sv-1]
            result = result > 0
            return result

        def h(sv, sw):
            return (h_vs(sv, sw) > 0) and (h_ws(sv, sw) > 0)

        # h_vs = lambda sv, sw: (1-sv*sw*l**2)*vbar[-sv] + l**2*sw*cumsum_vbar[sv-1] - l*cumsum_wbar[sw-1]
        # h_ws = lambda sv, sw: ((1-sv*sw*l**2)*wbar[-sw] + l**2*sv*cumsum_wbar[sw-1] - l*cumsum_vbar[sv-1]) > 0
        # h = lambda sv, sw: (h_vs(sv,sw) > 0) and (h_ws(sv,sw) > 0)
        candidates = binary_search_2D(h, 1, len(vbar), 1, len(wbar))

        f = h_path_mult(vbar, wbar, l)
        f_best = f(0, 0)
        v_best = 0
        w_best = 0

        for s_v, s_w in candidates:
            v = np.zeros(len(vbar))
            w = np.zeros(len(wbar))
            v[-1] = ((1 - l**2 * s_v * s_w) * vbar[-1] + l**2 * s_w * cumsum_vbar[s_v-1] - l * cumsum_wbar[s_w-1]) / (1 - l**2 * s_v * s_w)
            v[-s_v:-1] = vbar[-s_v:-1] - vbar[-1] + v[-1]
            w[-s_w:] = wbar[-s_w:] - l * np.sum(v)

            fval = f(v, w)
            if fval < f_best:
                v_best = np.copy(v)
                w_best = np.copy(w)
                f_best = fval

        if f(0, wbar) < f_best:
            v_best = np.zeros(len(vbar))
            w_best = wbar
            f_best = f(0, wbar)

        if f(vbar, 0) < f_best:
            v_best = vbar
            w_best = np.zeros(len(wbar))
            f_best = f(vbar, 0)

        inverse_permutation_v = np.arange(
                len(sorting_permutation_v))[np.argsort(sorting_permutation_v)]
        inverse_permutation_w = np.arange(
                len(sorting_permutation_w))[np.argsort(sorting_permutation_w)]
        v_best = signs_v * v_best[inverse_permutation_v]
        w_best = signs_w * w_best[inverse_permutation_w]
        return v_best, w_best

    def prox_one_neuron(
            self, vbar, wbar, l, s_v_min=1, s_v_max=-1, s_w_min=1, s_w_max=-1):
        """
        Prox for one hidden neuron

        Args:
            vbar: argument of prox
            wbar: argument of prox
            l: lambda
            s_v_min: minimum sparsity for v
            s_v_max: maximum sparsity for v
            s_w_min: minimum sparsity for w
            s_w_max: maximum sparsity for w
        """
        if s_v_max < 0:
            s_v_max = len(vbar)
        if s_w_max < 0:
            s_w_max = len(wbar)
        s_v_min = max(s_v_min, 1)
        s_w_min = max(s_w_min, 1)
        s_v_max = min(s_v_max, len(vbar))
        s_w_max = min(s_w_max, len(wbar))

        signs_v = np.sign(vbar)
        signs_w = np.sign(wbar)
        vbar = np.abs(vbar)
        wbar = np.abs(wbar)
        sorting_permutation_v = np.argsort(vbar)
        vbar = vbar[sorting_permutation_v]
        sorting_permutation_w = np.argsort(wbar)
        wbar = wbar[sorting_permutation_w]

        f = h_path_mult(vbar, wbar, l)
        # t = 1 + 1 / l**2 # sparsity threshold
        f_best = f(0, 0)
        v_best = 0
        w_best = 0
        w = np.zeros(len(wbar))

        cumsum_v = vbar[::-1].cumsum()
        cumsum_w = wbar[::-1].cumsum()
        for s_w in range(s_w_min, s_w_max+1):
            sum_w = cumsum_w[s_w-1]
            # s_v_max_ = min(s_v_max, t / s_w)
            v = np.zeros(len(vbar))
            for s_v in range(s_v_min, s_v_max + 1):
                sum_v = cumsum_v[s_v-1]
                update = (l**2 * s_w * sum_v - l * sum_w)
                update = update / (1 - l ** 2 * s_v * s_w)
                v[-s_v:] = vbar[-s_v:] + update
                w[-s_w:] = wbar[-s_w:] - l * np.sum(v)
                if w[-s_w] and v[-s_v] >= 0:
                    fval = f(v, w)
                    if fval < f_best:
                        v_best = np.copy(v)
                        w_best = np.copy(w)
                        f_best = fval
                # else:
                #    break

        if f(0, wbar) < f_best:
            v_best = np.zeros(len(vbar))
            w_best = wbar
            f_best = f(0, wbar)

        if f(vbar, 0) < f_best:
            v_best = vbar
            w_best = np.zeros(len(wbar))
            f_best = f(vbar, 0)

        inverse_permutation_v = np.arange(len(sorting_permutation_v))
        inverse_permutation_v = inverse_permutation_v[
                np.argsort(sorting_permutation_v)]

        inverse_permutation_w = np.arange(len(sorting_permutation_w))
        inverse_permutation_w = inverse_permutation_w[
                np.argsort(sorting_permutation_w)]

        v_best = signs_v * v_best[inverse_permutation_v]
        w_best = signs_w * w_best[inverse_permutation_w]

        return v_best, w_best

