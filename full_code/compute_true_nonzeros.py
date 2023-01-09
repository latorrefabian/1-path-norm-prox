import pdb
import os
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rsparse.utils import filename_to_params, get_input_output_size
from rsparse.lipschitz import OneLayerNetwork, ELU
from rsparse.utils import nonzero_params_percentage

from defaults import (
        MODEL_DATA_FILE, TEST_ERRORS_DATA_FILE, ITER_DATA_FILE)


def subset_models(models, epochs=20, batch_size=100, hidden=200):
    # remove unnecessary columns
    models = models[models['epochs'] == epochs]
    models = models[models['batch_size'] == batch_size]
    models = models[models['hidden'] == hidden]
    models = models[models['reg'] != 'linfprod']
    models = models[models['cuda'] == False]
    models.drop(
            ['epochs', 'batch_size', 'hidden', 'cuda'],
            axis='columns', inplace=True)
    return models


def pivot_error(errors, error=0.1):
    err_piv = errors.pivot(columns='eps')
    err_piv['error'] = err_piv['test_error'][0.0]
    rob_err = 'robust error ' + r'($\epsilon=' + str(error) + '$)'
    err_piv[rob_err] = err_piv['test_error'][error]
    err_piv = err_piv[['error', rob_err]]
    err_piv.columns = err_piv.columns.get_level_values(0)
    return err_piv, rob_err


def non_zeros(V, W):
    """
    V: n-by-p matrix
    W: n-by-m matrix
    m: input dimension
    n: number of hidden neurons
    p: output dimension
    
    Returns: number of effective non zero entries of the matrices when computing V^T\sigma(Wx)
    """
    for i in range(W.shape[0]):
        pdb.set_trace()
        w = W[i, :]
        v = V[i, :]
        if (v==0).all():
            W[i, :] = 0
        if (w==0).all():
            V[i, :] = 0

    return len(V.nonzero()) + len(W.nonzero())


def set_to_zeros(module):
    """
    V: n-by-p matrix
    W: n-by-m matrix
    m: input dimension
    n: number of hidden neurons
    p: output dimension
    
    Returns: number of effective non zero entries of the matrices when computing V^T\sigma(Wx)
    """
    V = module.W2.weight.t()
    W = module.W1.weight
    for i in range(W.shape[0]):
       #      zero_params += torch.isclose(
       #              param, torch.zeros_like(param)).sum().item()
        w = W[i, :]
        v = V[i, :]
        if torch.isclose(v, torch.zeros_like(v)).all().item():
            W[i, :] = 0.
        if torch.isclose(w, torch.zeros_like(w)).all().item():
            V[i, :] = 0
    module.W2.weight.data = V.t()
    module.W1.weight.data = W


def set_to_zeros(module):
    """
    V: n-by-p matrix
    W: n-by-m matrix
    m: input dimension
    n: number of hidden neurons
    p: output dimension
    
    Returns: number of effective non zero entries of the matrices when computing V^T\sigma(Wx)
    """
    V = module.W2.weight.t()
    W = module.W1.weight
    for i in range(W.shape[0]):
       #      zero_params += torch.isclose(
       #              param, torch.zeros_like(param)).sum().item()
        w = W[i, :]
        v = V[i, :]
        if torch.isclose(v, torch.zeros_like(v)).all().item():
            W[i, :] = 0.
        if torch.isclose(w, torch.zeros_like(w)).all().item():
            V[i, :] = 0
    module.W2.weight.data = V.t()
    module.W1.weight.data = W


def compute_true_zeros(models):
    models = subset_models(models)
    for f in models.index:
        params = filename_to_params(f)
        if params['reg'] == 'linfpath':
            pass
        else:
            continue
        weights_file = os.path.join('ckpts', f + '.pt')
        in_features, out_features = get_input_output_size(params['dataset'])
        network = OneLayerNetwork(
                in_features=in_features, hidden_features=params['hidden'],
                out_features=out_features, activation=ELU())
        network.load_state_dict(torch.load(weights_file))
        original_nnz = nonzero_params_percentage(network)
        set_to_zeros(network)
        true_nnz = nonzero_params_percentage(network)
        nnz = models.loc[f]['nnz']
        difference = max(original_nnz - true_nnz, true_nnz - original_nnz)
        if difference > 1:
            pdb.set_trace()


def main(**kwargs):
    # pdb.set_trace()
    models = pd.read_pickle(MODEL_DATA_FILE)
    models.loc[models['reg'] == 'l1', 'reg'] = r'$\ell_1$'
    models.loc[models['reg'] == 'linfpath', 'reg'] = 'path'
    models.loc[models['reg'] == 'linfproj', 'reg'] = 'layer'
    models.loc[models['reg'] == 'layer', 'lambda_'] = (
            models.loc[models['reg'] == 'layer', 'lambda_'].apply(np.reciprocal))

#     errors = pd.read_pickle(TEST_ERRORS_DATA_FILE).set_index('file')
    #iters = pd.read_pickle(ITER_DATA_FILE).set_index('file')

#     robust_vs_error_plot(
#             models.copy(deep=True), errors.copy(deep=True), error=0.1)
#     error_vs_lambda_plot(
#             models.copy(deep=True), errors.copy(deep=True), error=0.1)
    compute_true_zeros(models.copy(deep=True))
#     sparsity_plot(
#             models.copy(deep=True), iters.copy(deep=True))
    # mnist_lambda_lr_path_nnz(models.copy(deep=True), iters.copy(deep=True))
#     loss_vs_lr(
#             models.copy(deep=True), errors.copy(deep=True), error=0.1)

    #nnz_vs_robust_plot = sns.scatterplot(
    #        x='nnz', y='robust error', hue='reg', style='prox', data=model_errors)
    #nnz_vs_robust_plot.get_figure().savefig('nnz_vs_robust.pdf')

    #nnz_vs_robust_plot2 = sns.relplot(
    #        x='nnz', y='robust error', col='reg', hue='lambda_', style='prox', data=model_errors)
    #nnz_vs_robust_plot2.savefig('nnz_vs_robust2.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))

