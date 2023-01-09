from torch.optim import SGD


class ProxSGD:
    """
    Proximal Stochastic Gradient Descent optimizer

    Args:
        module (nn.Module): module whose parameters will be optimized
        lr (float): learning rate (step size)
        regularizer (rsparse.regularizer): proximable regularizer
        lambda_ (float): penalty parameter
        **kwargs: extra parameters passed to torch.optim.SGD
    """
    def __init__(self, module, lr, regularizer, lambda_, **kwargs):
        self.module = module
        self.lr = lr
        self.optimizer = SGD(
                module.parameters(), lr=lr, momentum=0.0, **kwargs)
        self.regularizer = regularizer
        self.lambda_ = lambda_

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform one optimization step, composed of one stochastic gradient step
        and one proximal step
        """
        self.optimizer.step()
        self.regularizer.prox(self.lr * self.lambda_, self.module)


class ProxSGD2(SGD):
    """
    Alternative implementation of Proximal Stochastic Gradient Descent
    optimizer

    Args:
        params (iterable): parameters that will be optimized
        lambda_ (float): penalty parameter
        prox (callable): proximal operator. Will be called once after each
            stochastic gradient update.
        lr (float): learning rate
        **kwargs: extra parameters passed to torch.optim.SGD

    """
    def __init__(self, params, lambda_, prox, lr, **kwargs):
        super().__init__(params, lr, **kwargs)
        self.lambda_ = lambda_
        self.prox = prox
        self.lr = lr

    def step(self, closure=None):
        super().step(closure=closure)
        self.prox(self.lr * self.lambda_)

