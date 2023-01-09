import torch


def soft_thresh(lambda_, x):
    """
    Soft thresholding operator

    Args:
        lambda_ (float): soft-threshold value
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - lambda_, min=0.)


def proj_simplex(x, r):
    """
    """
    idx = torch.sum(x, dim=1) > r
    x_ = x[idx]
    n, m = x_.shape

    if n == 0:
        return x

    s = torch.sort(x_, dim=1, descending=True)[0]
    rng_ = torch.arange(m, device=s.device)
    rng = (rng_.float() + 1.).repeat(n, 1)
    cs = (torch.cumsum(s, dim=1) - r) / rng
    st = s - cs
    theta = torch.max((st > 0).long() * rng_, dim=1)[0]
    lambda_ = torch.gather(cs, dim=1, index=theta[:, None])
    x[idx] = torch.clamp(x_ - lambda_, min=0.)
    return x


def proj_l1(x, r):
    """
    Projection onto an L1 ball with radius r.

    Args:
        x (torch.tensor): rank-two tensor (matrix) whose rows will be
            projected onto the L1-norm ball of certain radius.
        r (float): radius of the L1-norm ball.

    References:
        J. Duchi, S. Shalev-Shwartz, and Y. Singer, "Efficient projections onto
        the l1-ball for learning in high dimensions" 2008.

    """
    return torch.sign(x) * proj_simplex(torch.abs(x), r=r)


def proj_2_matrix_norm(x, r):
    """
    Projection onto the 2-matrix-norm ball with radius r. This
    operation is equivalent to truncation of the singular values of the
    matrix x, with threshold r.

    Args:
        x (torch.tensor): rank-two tensor (matrix) that will be
            projected onto the 2-matrix-norm ball of certain radius.
        r (float): radius of the 2-matrix-norm ball.

    """
    if x.shape[0] == 1:
        norm = torch.norm(x, p=2)
        if norm.item() > r:
            x = x / norm
        return x

    u, s, v = torch.svd(x, some=True)

    if torch.max(s).item() < r:
        return

    torch.clamp_(s, max=r)
    x = u @ torch.diag(s) @ v.t()
    return x


def proj_inf_matrix_norm(x, r):
    """
    Projection onto the infinity-matrix-norm ball with radius r. This
    operation is equivalent to projection onto the L1-norm ball with radius r,
    of the rows of matrix x.

    Args:
        x (torch.tensor): rank-two tensor (matrix) that will be
            projected onto the infinity-matrix-norm ball of certain radius.
        r (float): radius of the infinity-matrix-norm ball.

    """
    return proj_l1(x, r=r)

