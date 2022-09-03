import torch
import numpy as np
from typing import Callable, List, Tuple


def binary_search_2D(
        h: Callable[[int, int], float],
        start_i: int,
        end_i: int,
        start_j: int,
        end_j: int) -> List[Tuple[int, int]]:
    """
    2D efficient search procedure for 2D sorted array
    Returns a list of candidates
    """
    i = start_i
    j = end_j
    candidates = []
    can_add = True
    while i <= end_i and j >= start_j:
        val = h(i, j)
        if val <= 0:
            if i > start_i and can_add:
                candidates.append([i-1, j])
                can_add = False  # until we increase i
            j -= 1
        else:
            i += 1
            can_add = True

    if i == end_i + 1:
        candidates.append([i-1, j])

    return candidates


def h_path_mult(
        x: np.ndarray,
        y: np.ndarray,
        l: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Objective to optimize in the prox with multiple outputs
    """
    def h(v, w):
        ans = 0.5 * np.linalg.norm(v - x) ** 2
        ans += 0.5 * np.linalg.norm(w - y) ** 2
        ans += l * np.sum(np.tensordot(np.abs(v), np.abs(w), axes=0))
        return ans

    return h


def prox_one_neuron_fast(
        vbar: np.ndarray,
        wbar: np.ndarray,
        lambda_: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prox for one hidden neuron

    Args:
        vbar: argument of prox
        wbar: argument of prox
        l: lambda
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
        result = (1 - sv * sw * lambda_ ** 2) * vbar[-sv]
        result += lambda_ ** 2 * sw * cumsum_vbar[sv-1]
        result += - lambda_ * cumsum_wbar[sw-1]
        return result

    def h_ws(sv, sw):
        result = (1 - sv * sw * lambda_ ** 2) * wbar[-sw]
        result += lambda_ ** 2 * sv * cumsum_wbar[sw-1]
        result += - lambda_ * cumsum_vbar[sv-1]
        result = result > 0
        return result

    def h(sv, sw):
        return (h_vs(sv, sw) > 0) and (h_ws(sv, sw) > 0)

    candidates = binary_search_2D(
        h=h,
        start_i=1,
        end_i=len(vbar),
        start_j=1,
        end_j=len(wbar))

    f = h_path_mult(vbar, wbar, lambda_)
    v_best = np.zeros(len(vbar))
    w_best = np.zeros(len(wbar))
    f_best = f(v_best, w_best)

    for s_v, s_w in candidates:
        v = np.zeros(len(vbar))
        w = np.zeros(len(wbar))
        _value = (1 - lambda_ ** 2 * s_v * s_w) * vbar[-1]
        _value += lambda_ ** 2 * s_w * cumsum_vbar[s_v-1]
        _value -= lambda_ * cumsum_wbar[s_w-1]
        v[-1] = _value / (1 - lambda_ ** 2 * s_v * s_w)
        v[-s_v:-1] = vbar[-s_v:-1] - vbar[-1] + v[-1]
        w[-s_w:] = wbar[-s_w:] - lambda_ * np.sum(v)

        fval = f(v, w)
        if fval < f_best:
            v_best = np.copy(v)
            w_best = np.copy(w)
            f_best = fval

    if f(np.zeros(len(vbar)), wbar) < f_best:
        v_best = np.zeros(len(vbar))
        w_best = wbar
        f_best = f(v_best, wbar)

    if f(vbar, np.zeros(len(wbar))) < f_best:
        v_best = vbar
        w_best = np.zeros(len(wbar))
        f_best = f(vbar, w_best)

    inverse_permutation_v = np.arange(
            len(sorting_permutation_v))[np.argsort(sorting_permutation_v)]
    inverse_permutation_w = np.arange(
            len(sorting_permutation_w))[np.argsort(sorting_permutation_w)]
    v_best = signs_v * v_best[inverse_permutation_v]
    w_best = signs_w * w_best[inverse_permutation_w]
    return v_best, w_best


def prox_full_fast(
        vbar: torch.Tensor,
        wbar: torch.Tensor,
        lambda_: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prox of the path-norm f(v, w) = 1^T|v||w|1
    that means it is a solution of the problem
    min_{x, y} 0.5 ||x - v||^2 + 0.5 ||y - w||^2 + lambda_ 1^T|x||y|1
    """
    v = torch.zeros(vbar.shape)
    w = torch.zeros(wbar.shape)
    h = vbar.shape[1]

    # For loop over hidden neurons
    for i in range(h):
        v_, w_ = prox_one_neuron_fast(
            np.array(vbar[:, i]), np.array(wbar[i, :]), lambda_)
        v[:, i] = torch.tensor(v_)
        w[i, :] = torch.tensor(w_)

    return v, w


if __name__ == '__main__':
    test_v = torch.randn(5, 3)
    test_w = torch.randn(3, 10)
    lambda_ = 0.001
    prox_v, prox_w = prox_full_fast(
        vbar=test_v,
        wbar=test_w,
        lambda_=lambda_)
    _ = 0.0
