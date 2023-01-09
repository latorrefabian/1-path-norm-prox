import math
import torch


def objective_factory(x, Y, lambda_):
    def objective(v, W):
        pass

    return objective


def prox_positive_factory(m, lambda_, device):
    """
    Computes the function that performs the following:

    For x in R_+, Y in R^m_+ return the minimizer of

    f(v, W) = || (v, W) - (x, Y) ||^2 + lambda_ * || diag(v) W ||_1

    """
    s_max = math.floor(1 + lambda_ ** (-2))
    s_max = max(s_max, m)
    k = _k_vector(s_max, lambda_, device)
    k_matrix = _k_matrix(k)

    def prox(x, Y):
        """
        The actual function
        """
        ord_Y = torch.topk(Y, k=s_max, dim=1, sorted=True)[0]
        rho = ord_Y * k
        rho = torch.cat((x[:, None], rho), dim=1)
        v = k_matrix * rho[:, None, :]
        v = torch.sum(v, dim=2)
        v = torch.clamp(v, min=0.)
        W = - lambda_ * v + Y
        W = torch.clamp(W, min=0.)
        obj = objective(v, W, lambda_)
        argmin = torch.min(obj, dim=1)[1]
        return v[argmin], Y[argmin]

    return prox


def _k_vector(s_max, lambda_, device):
    """
    Compute the vector (1 + 1 / (1 - lambda_ ** 2 * s)) for s between 0 and
    s_max. This is used in the computation of the path-norm proximal map.

    """
    k = torch.arange(0., s_max + 1, device=device)
    k.mul_(- lambda_ ** 2)
    k.add_(1.)
    k.reciprocal_()
    k.add_(1.)
    k[0] = 1.
    return k


def _k_matrix(k):
    """
    given a vector k in R^n, compute the matrix (illustrated here when n=4),
    so the vector k has components (k_0, k_1, k_2, k_3)

    1           | 0         | 0    | 0
    k_1         | -1        | 0    | 0
    k_2 k_1     | -k_2      | -1   | 0
    k_3 k_2 k_1 | -k_3 k_2  | -k_3 | -1

    """
    cp = torch.cumprod(k, dim=0)
    mat = cp[:, None].repeat(1, len(cp)) / cp
    mat.tril_()
    mat[:, 1:].mul_(-1.)
    return mat

