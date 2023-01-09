"""
Script to train a neural network and save to disk weights, on a particular fold
of a cross-validation split
"""
import argparse
import defaults
import torch
import os
import pickle

from callback import Callback
from torch import optim

from rsparse.regularizer import L1, PathLength2
from rsparse.utils import (
        train, regularized_cross_entropy,
        get_dataset_cv, get_input_output_size,
        params_to_filename, test_error)
from rsparse.lipschitz import OneLayerNetwork, ThreeLayerNetwork, ELU
from rsparse.optim import ProxSGD2 as ProxSGD


class LinfLipschitzIndicator:
    """
    Layer-wise constrain on the Lipschitz constant with respect to the
    L-infinity norm. When prox is True, it is equivalent to projected gradient
    descent and each layer is constrained to have Lipschitz constant at most
    lambda_. If prox is False, then a penalty on the sum of layerwise Lipschitz
    constants is added to the loss, and lambda_ becomes the penalty parameter.

    This is slightly different from the other regularizers like L1 or path,
    where the loss function is the same whether prox is True or False. The idea
    here is that adding a layerwise penalty on the Lipschitz constant of each
    layer, is equivalent to constrained optimization. Notice that the same
    penalty lambda_ is used for each layer of the network.

    """
    @staticmethod
    def prox_factory(module):
        def prox_(lambda_):
            module.constrain(r=lambda_, p=float('inf'))

        return prox_

    @staticmethod
    def loss_factory(prox):
        if prox:
            reg = None
        else:
            def reg(module):
                return sum(module.layerwise_lipschitz(p=float('inf')))

        def loss_(lambda_):
            return regularized_cross_entropy(lambda_, reg=reg)

        return loss_


class LinfLipschitzPath:
    """
    Penalty on the Path-norm of a neural network, with respect to the
    L-infinity norm.

    """
    @staticmethod
    def prox_factory(module):
        reg = PathLength2(requires_grad=False)

        def prox_(lambda_):
            reg.prox(lambda_, module)

        return prox_

    @staticmethod
    def loss_factory(prox):
        if prox:
            def reg(module):
                with torch.no_grad():
                    return module.lipschitz(
                            method='path', p=float('inf')).detach()
        else:
            def reg(module):
                return module.lipschitz(method='path', p=float('inf'))

        def loss_(lambda_):
            return regularized_cross_entropy(lambda_, reg=reg)

        return loss_


class LinfLipschitzProd:
    """
    Penalty on the upper bound of the L-infinity-norm Lipschitz constant of a
    neural network, given by the product of the layer-wise constants. Proximal
    mapping is not implemented so can only be used with subgradient methods.

    """
    @staticmethod
    def prox_factory(module):
        def prox_(lambda_):
            raise NotImplementedError('product prox not implemented')

        return prox_

    @staticmethod
    def loss_factory(prox):
        if prox:
            raise ValueError(
                    'regularizer based on product of layer-wise '
                    'constants has no proximal mapping yet')
        else:
            def reg(module):
                return module.lipschitz(method='product', p=float('inf'))

        def loss_(lambda_):
            return regularized_cross_entropy(lambda_, reg=reg)

        return loss_


class L1Norm:
    """
    Penalty on the L1-norm of the concatenation of all the parameters of the
    network. The Proximal operator is just soft-thresholding.

    """
    @staticmethod
    def prox_factory(module):
        reg = L1(bias=False, requires_grad=False)

        def prox_(lambda_):
            reg.prox(lambda_, module)

        return prox_

    @staticmethod
    def loss_factory(prox):
        reg = L1(bias=False, requires_grad=not prox)

        def loss_(lambda_):
            return regularized_cross_entropy(lambda_, reg)

        return loss_


def reg_factory(reg):
    """
    Given a regularization code name, return an instance of the class
    that contains the methods creating the regularized loss function and the
    proximal operator of the regularizer.

    """
    if reg == 'linfproj':
        return LinfLipschitzIndicator()
    elif reg == 'linfpath':
        return LinfLipschitzPath()
    elif reg == 'linfprod':
        return LinfLipschitzProd()
    elif reg == 'l1':
        return L1Norm()
    else:
        raise ValueError('unsupported regularization ' + str(reg))


def main(
        reg, batch_size, lr, seed, epochs, prox,
        dataset, hidden, lambda_, cuda, n_folds, fold, folds_dir,
        ckpt_dir, errors_dir, overwrite=False):
    """
    Train a network on a kfold of data and save weights to disk
    The parameters 'reg', 'batch_size', 'lr', 'seed', 'epochs', 'prox',
    'dataset', 'hidden', 'lambda_', 'cuda', 'n_folds', 'fold' are
    described in the 'help_' object in the 'defaults.py' file.

    Args:
        ckpt_dir (str): directory where the weights of the trained networks
            will be stored.
        folds_dir (str): directory where the indexes of the folds for
            crossvalidation are stored.
        overwrite (bool): if True, the module will be retrained and saved even
            if a previous version with the same parameters exists. In that
            case, the old file will be overwritten.

    """
    if cuda:
        assert torch.cuda.is_available()

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(errors_dir):
        os.mkdir(errors_dir)

    filename_ = params_to_filename(
            reg=reg, batch_size=batch_size, lr=lr, seed=seed, epochs=epochs,
            prox=prox, dataset=dataset, hidden=hidden, lambda_=lambda_,
            cuda=cuda, n_folds=n_folds, fold=fold)
    ckpt = os.path.join(ckpt_dir, filename_) + '.pt'
    input_, output_ = get_input_output_size(name=dataset)
    if len(hidden) == 1:
        module = OneLayerNetwork(input_, hidden[0], output_, activation=ELU())
    elif len(hidden) == 3:
        module = ThreeLayerNetwork(input_, hidden, output_, activation=ELU())
    train_loader, test_loader = get_dataset_cv(
            folds_dir=folds_dir, name=dataset, batch_size=batch_size,
            n_folds=n_folds, fold=fold)
    if reg == 'linfprod':
        if len(hidden) == 1:
            lambda_ = lambda_ ** (1 / 2)
        elif len(hidden) == 3:
            lambda_ = lambda_ ** (1 / 4)
    elif reg == 'linfpath':
        if len(hidden) == 3:
            lambda_ = lambda_ ** (1 / 2)

    if os.path.isfile(ckpt) and not overwrite:
        print('A model with the same configuration already exists.')
        module.load_state_dict(torch.load(ckpt))
        device = torch.device('cuda:0' if cuda else 'cpu')
        torch.manual_seed(seed)
        module.to(device)
    else:
        device = torch.device('cuda:0' if cuda else 'cpu')
        torch.manual_seed(seed)
        module.to(device)
        reg_ = reg_factory(reg)
        loss_ = reg_.loss_factory(prox)

        if prox:
            optimizer = ProxSGD(
                    module.parameters(), lr=lr, lambda_=lambda_,
                    prox=reg_.prox_factory(module))
        else:
            optimizer = optim.SGD(module.parameters(), lr=lr)

        def loss(**kwargs):
            return kwargs['loss_'].item()

        iterations = int(len(train_loader) * epochs)
        if iterations > 1000:
            skip = int(iterations / 1000)
        else:
            skip = 1

        cb = Callback(loss, skip=skip)
        loss_fn = loss_(lambda_)

        train(
                module=module, loss=loss_fn, optimizer=optimizer,
                train_loader=train_loader, epochs=epochs, callback=cb)

        torch.save(module.state_dict(), ckpt)

    print('evaluating model')
    f_wo_ext = os.path.join(errors_dir, filename_)
    outfile = f_wo_ext + '.pkl'
    epsilons = defaults.epsilons[dataset]

    entries = list()
    for eps in epsilons:
        entry = {'file': filename_, 'eps': eps}
        entry['test_error'] = test_error(
                module=module, test_loader=test_loader, eps=eps)
        print('epsilon: ', eps, ' error: ', entry['test_error'])
        entries.append(entry)

    with open(outfile, 'wb') as handle:
        pickle.dump(entries, file=handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--seed', type=int, default=2, help=defaults.help_['seed'])
    parser.add_argument(
            '--batch_size', type=int, default=100,
            help=defaults.help_['batch_size'])
    parser.add_argument(
            '--reg', type=str, help=defaults.help_['reg'])
    parser.add_argument(
            '--epochs', type=int, default=10, help=defaults.help_['epochs'])
    parser.add_argument(
            '--lr', type=float, default=2e-1, help=defaults.help_['lr'])
    parser.add_argument(
            '--prox', action='store_true', help=defaults.help_['hidden'])
    parser.add_argument(
            '--dataset', type=str, default='mnist',
            help=defaults.help_['dataset'])
    parser.add_argument(
            '--hidden', type=int, nargs='+', default=[200],
            help=defaults.help_['hidden'])
    parser.add_argument(
            '--ckpt_dir', type=str, default=defaults.CV_CKPT_DIR,
            help=defaults.help_['ckpt_dir'])
    parser.add_argument(
            '--folds_dir', type=str, default=defaults.FOLDS_DIR,
            help=defaults.help_['folds_dir'])
    parser.add_argument(
            '--n_folds', type=int, default=4,
            help=defaults.help_['n_folds'])
    parser.add_argument(
            '--fold', type=int, default=0,
            help=defaults.help_['fold'])
    parser.add_argument(
            '--cuda', action='store_true', help=defaults.help_['cuda'])
    parser.add_argument(
            '--lambda_', type=float, default=0.,
            help=defaults.help_['lambda_'])
    parser.add_argument(
            '--overwrite', action='store_true',
            help=defaults.help_['overwrite'])
    parser.add_argument(
            '--errors_dir', type=str, default=defaults.CV_ERRORS_DIR,
            help=defaults.help_['errors_dir'])

    args = parser.parse_args()
    print(args)
    main(**vars(args))
    print('\nDone')

