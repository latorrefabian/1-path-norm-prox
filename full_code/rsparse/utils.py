# import pdb
import ast
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torchvision

from advertorch.attacks import LinfPGDAttack
from sklearn.model_selection import KFold
from torch.utils.data import Subset

from torch import nn
from tqdm import tqdm


SEPARATOR = ')('  # used for separating key-value pairs in filenames


def train(
        module, loss, optimizer, train_loader, epochs,
        callback=None):
    """
    Module training loop

    Args:
        module (nn.Module): module whose parameters will be optimized
        loss (callable): objective function to minimize
        optimizer (Optimizer): optimizer of the module weights
        train_loader (torch.utils.data.DataLoader): training data loader
        epochs (int): number of epochs (passes over the dataset)
        callback (callable): used to keep per iteration information about
            training

    """
    module.train()
    device = next(module.parameters()).device
    iterations = int(len(train_loader) * epochs)
    data_sampler = sampler(train_loader, iterations)
    pbar = tqdm(enumerate(data_sampler), total=iterations)

    for i, (x, y) in pbar:
        optimizer.zero_grad()
        loss_ = loss(module, x.to(device), y.to(device))

        if callback is not None:
            callback(locals())
            pbar.set_description(str(callback))

        loss_.backward()
        optimizer.step()


def _test_error(module, test_loader):
    """
    Evaluate a model on test data

    Args:
        module (nn.Module): model for which to compute the test error
        test_loader (torch.utils.data.DataLoader): test data loader

    Returns:
        (float): value of the loss in the test data
        (float): accuracy on the test data

    """
    device = next(module.parameters()).device
    correct = 0.

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = module(x).data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()

    accuracy = correct.item() / len(test_loader.dataset) * 100
    return 100. - accuracy


def test_error(
        module, test_loader, eps=0.0, nb_iter=40, eps_iter=None,
        clip_min=0.0, clip_max=1.0):
    """
    Evaluate a model on adversarial examples, constrained with the
    L_infinity-norm. If the perturbation size is set to zero, this is
    equivalent to the usual test set error.

    Args:
        module (nn.Module): model for which to compute the robust test error
        test_loader (torch.utils.data.DataLoader): test data loader
        eps (float): maximum size of perturbation. If set to zero it will
            evaluate the performance on unperturbed test samples.
        nb_iter (int): number of iterations
        eps_iter (float): step size

    Returns:
        (float): robust test error as a percentage

    """
    if eps == 0.:
        return _test_error(module, test_loader)

    if eps_iter is None:
        eps_iter = 2 * eps / nb_iter

    adversary = LinfPGDAttack(
        module, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps,
        nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True,
        clip_min=clip_min, clip_max=clip_max, targeted=False)

    correct = 0.

    for x, y in test_loader:
        adv = adversary.perturb(x, y)
        pred = module(adv).data.max(1, keepdim=True)[1]
        correct += pred.eq(y.data.view_as(pred)).sum()

    accuracy = correct.item() / len(test_loader.dataset) * 100
    return 100. - accuracy


def sampler(loader, iterations):
    """
    Epoch sampler

    Args:
        loader (torch.utils.data.DataLoader): data iterator
        iterations (int): number of iterations

    Yields:
        (object): same type as 'loader' yield value

    """
    i = 0
    finished = False
    while not finished:
        for x in loader:
            if i >= iterations:
                finished = True
                break
            i += 1
            yield x


def true_nonzero_params_percentage(module=None):
    """
    Compute the sparsity of a module by computing the number of zeros and
    total entries in weight parameters.

    Args:
        module (torch.nn.Module): module for which sparsity will be computed

    Returns:
        (float): sparsity percentage, higher is better.

    """
    total_params = 0.
    non_zero_params = 0.

    V = module.W2.weight.t()
    W = module.W1.weight
    V_bias = module.W2.bias
    W_bias = module.W1.bias

#     V = torch.tensor([
#         [1., 0, 1.],
#         [0., 0, 0.],
#         [1., 0, 1.],
#         [1., 0, 1.],
#         ])
#     W = torch.tensor([
#         [0., 0, 0.],
#         [0., 1., 0.],
#         [1., 0, 1.],
#         [0., 0, 0.],
#         ])
#
#     V_bias = torch.tensor([0., 0, 0., 0])
#     W_bias = torch.tensor([0., 0, 0., 0])

    total_params += V.numel() + W.numel()
    total_params += V_bias.numel() + W_bias.numel()

    V_iszero = torch.isclose(V, torch.zeros_like(V)).float()
    Vprod = 1. - torch.prod(V_iszero, dim=1)

    W_iszero = torch.isclose(W, torch.zeros_like(W)).float()
    Wprod = 1. - torch.prod(W_iszero, dim=1)

    V = V * Wprod[:, None]
    W = W * Vprod[:, None]

    non_zero_params += len(V.nonzero())
    non_zero_params += len(W.nonzero())
    non_zero_params += len(V_bias) - torch.sum(
            torch.isclose(V_bias, torch.zeros_like(V_bias))).item()
    non_zero_params += len(W_bias) - torch.sum(
            torch.isclose(W_bias, torch.zeros_like(W_bias))).item()

    return non_zero_params / total_params * 100


def nonzero_params_percentage(module):
    """
    Compute the sparsity of a module by computing the number of zeros and
    total entries in weight parameters.

    Args:
        module (torch.nn.Module): module for which sparsity will be computed

    Returns:
        (float): sparsity percentage, higher is better.

    """
    total_params = 0.
    zero_params = 0.

    for name, param in module.named_parameters():
        if 'weight' in name or 'bias' in name:
            total_params += param.data.numel()
            zero_params += torch.isclose(
                    param, torch.zeros_like(param)).sum().item()

    return 100 - zero_params / total_params * 100


def regularized_cross_entropy(lambda_, reg=None, fgsm=0.0):
    """
    Regularized Cross Entropy Loss

    Args:
        lambda_ (float): penalty parameter
        reg (Regularizer): regularizer
        fgsm (float): bound on adversarial perturbation for adversarial
            training. If set to zero then no adversarial training is used

    Returns:
        (callable): function that given a module and an (x, y) batch of data,
            computes the regularized loss

    """
    if reg is None or lambda_ == 0.:
        def reg(module):
            return 0.

    ce = nn.CrossEntropyLoss()

    def loss(module, x, y):
        if fgsm > 0.0:
            delta = torch.zeros_like(x).uniform_(-fgsm, fgsm)
            delta.requires_grad = True
            output = module(x + delta)
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            delta.data = torch.clamp(
                    delta + 1.25 * fgsm * torch.sign(grad), -fgsm, fgsm)
            delta.data = torch.max(torch.min(1-x, delta.data), 0-x)
            delta = delta.detach()
            value = ce(module(torch.clamp(x + delta, 0, 1)), y)
        else:
            value = ce(module(x), y)
        value += lambda_ * reg(module)
        return value

    return loss


def get_data(name):
    if name == 'mnist':
        train = torchvision.datasets.MNIST(
                '~/.data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
        test = torchvision.datasets.MNIST(
                '~/.data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
    elif name == 'fmnist':
        train = torchvision.datasets.FashionMNIST(
                '~/.data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))

        test = torchvision.datasets.FashionMNIST(
                '~/.data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
    elif name == 'kmnist':
        train = torchvision.datasets.KMNIST(
                '~/.data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))

        test = torchvision.datasets.KMNIST(
                '~/.data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
    elif name == 'emnist':
        train = torchvision.datasets.EMNIST(
                '~/.data', split='balanced', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))

        test = torchvision.datasets.EMNIST(
                '~/.data', split='balanced', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
    elif name == 'cifar10':
        train = torchvision.datasets.CIFAR10(
                '~/.data', train=True, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))

        test = torchvision.datasets.CIFAR10(
                '~/.data', train=False, download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ]))
    else:
        raise ValueError('unsupported dataset ' + str(name))

    return train, test


def get_dataset_cv(
        folds_dir, name, batch_size, n_folds, fold):
    """
    Get train and test loaders for a dataset, in a way that is compatible
    with the training loop.

    Args:
        folds_dir (str): directory where the indices of the folds are stored
        name (str): given name of a dataset e.g., "mnist", currently the only
           implemented but should add more
        batch_size (int): size of data batches that should be used for the
            training set loader. The batch size for the test set is fixed at
            1000 samples.
        n_folds (int): number of folds for crossvalidation
        fold (int): index of the fold

    Returns:
        (torch.utils.data.DataLoader): training data loader
        (torch.utils.data.DataLoader): test data loader

    """
    train, test = get_data(name)
    filename = fold_filename(
            folder=folds_dir, n_splits=n_folds, fold=fold, dataset_name=name)

    with open(filename, 'rb') as f:
        index = pickle.load(f)

    train_set = Subset(train, index['train'])
    test_set = Subset(train, index['test'])

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=1000, shuffle=True)

    return train_loader, test_loader


def get_dataset(name, batch_size):
    """
    Get train and test loaders for a dataset, in a way that is compatible
    with the training loop.

    Args:
        name (str): given name of a dataset e.g., "mnist", currently the only
           implemented but should add more
        batch_size (int): size of data batches that should be used for the
            training set loader. The batch size for the test set is fixed at
            1000 samples.

    Returns:
        (torch.utils.data.DataLoader): training data loader
        (torch.utils.data.DataLoader): test data loader

    """
    train, test = get_data(name)
    train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
            test, batch_size=1000, shuffle=True)
    return train_loader, test_loader


def h_path_mult(x, y, l):
    """
    Objective to optimize in the prox with multiple outputs
    """
    def h(v, w):
        ans = 0.5 * np.linalg.norm(v - x) ** 2
        ans += 0.5 * np.linalg.norm(w - y) ** 2
        ans += l * np.sum(np.tensordot(np.abs(v), np.abs(w), axes=0))
        return ans
    return h


def h_path(lambda_, x, y):
    def h(v, w):
        ans = 0.5 * (v - x) ** 2 + 0.5 * torch.norm(w - y) ** 2
        ans += lambda_ * v * torch.sum(w)
        return ans

    return h


def get_input_output_size(name):
    """
    Obtain the input and output size (number of classes) of a classifier
    for a given dataset

    Args:
        name (str): given name of dataset

    Returns:
        (int): size of input as a flat vector, i.e., mnist input size is 784
        (int): size of output (number of classes)

    """
    if name == 'mnist':
        return 784, 10
    if name == 'fmnist':
        return 784, 10
    if name == 'kmnist':
        return 784, 10
    if name == 'emnist':
        return 784, 47
    if name == 'cifar10':
        return 3072, 10
    else:
        raise NotImplementedError(
                'unknown input-output size for dataset ' + str(name))


def params_to_filename(**kwargs):
    """
    Transform keyword - argument combinations to a string ID to use as filename

    Args:
        kwargs: keyword-value pairs. Values will be casted as strings.

    Returns:
        (str): string containing all keyword - value pairs. The keyword and
            value pairs appear sequentially separated by the SEPARATOR string.

    Raises:
        (ValueError): String constant SEPARATOR is used so that the
        operation can be reversed i.e., we can recover the parameters given a
        filename. For this reason the keyword and value of the argument should
        not contain the SEPARATOR value as substrings.
        (ValueError): if no keyword parameters are passed as arguments.

    """
    if len(kwargs) == 0:
        raise ValueError('should pass at least one keyword parameter')

    id_ = ''

    for i, (k, v) in enumerate(sorted(kwargs.items())):
        if SEPARATOR in k:
            raise ValueError(
                    'parameter name ' + k + 'contains substrings '
                    'reserved for SEPARATOR constant.')
        if SEPARATOR in str(v):
            raise ValueError(
                    'value ' + str(v) + 'contains substrings '
                    'reserved for SEPARATOR constant.')

        id_ += k + SEPARATOR + str(v)

        if i < len(kwargs) - 1:
            id_ += SEPARATOR

    return id_


def filename_to_params(filename):
    """
    Given a filename obtained as the output of a function with keyword
    parameters, recover the parameters used. This is inverse function
    to 'params_to_filename'.

    Args:
        filename (str): path to a filename generated with the
            'params_to_filename' function

    Returns:
        (dict): keyword-value pairs used to generate the filename

    """
    filename = os.path.basename(filename)
    components = filename.split(SEPARATOR)
    params = dict()

    i = 0
    while i < len(components):
        key, value = components[i], components[i+1]
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        elif value.isdigit():
            value = int(value)
        elif '[' in value:
            value = ast.literal_eval(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key] = value
        i += 2

    return params


def remove_extension(path):
    """
    Removes the extension e.g., '.pt', '.exe', etc. from a filename
    """
    return os.path.splitext(path)[0]


def params_dict_to_str(x):
    """
    Turn a dictionary into a string to be passed as options to a python script
    that uses an argparse.ArgumentParser()

    Args:
        x (dict): parameter name - value pairs

    Returns:
        (str): parameter names are preceded by '--' and the values are
            separated by a space.

    """
    params = list()

    for k, v in x.items():
        if type(v) is bool:
            if v:
                params.append('--' + k)
        elif type(v) is list:
            v_str = ' '.join([str(x) for x in v])
            params.append('--' + k + ' ' + v_str)
        else:
            params.append('--' + k + ' ' + str(v))

    return ' '.join(params)


def binary_search_2D(h, start_i, end_i, start_j, end_j):
    """
    2D efficient search procedure for 2D sorted arry
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
    if i == end_i+1:
        candidates.append([i-1, j])
    return candidates


def fold_filename(folder, n_splits, fold, dataset_name):
    """
    Returns the filename of a particular fold for cross-validation
    """
    return os.path.join(
            folder, dataset_name, str(n_splits) + '_folds', str(fold) + '.pkl')


def save_folds(folder, n_splits, dataset, dataset_name):
    """
    Write to disk indices of the folds for a given dataset for cross-validation
    """
    folder = os.path.expanduser(folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    dataset_folder = os.path.join(folder, dataset_name)
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)

    folds_folder = os.path.join(dataset_folder, str(n_splits) + '_folds')
    if not os.path.exists(folds_folder):
        os.mkdir(folds_folder)

    kf = KFold(n_splits=n_splits)
    index = np.arange(len(dataset))

    for i, (train_index, test_index) in enumerate(kf.split(index)):
        filename = fold_filename(folder, n_splits, i, dataset_name)
        with open(filename, 'wb') as f:
            indices = {'train': train_index, 'test': test_index}
            pickle.dump(indices, f, pickle.HIGHEST_PROTOCOL)

