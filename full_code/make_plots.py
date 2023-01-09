import pdb
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

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


def robust_vs_error_plot(
        models, errors, epochs=20, batch_size=100, hidden=200, error=0.1):
    """
    Plots the robust error vs error tradeoff for different regularizers and
    datasets

    """
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)
    models = models[
            (models['prox'] == True) & (models['reg'] == r'$\ell_1$')
            | ((models['prox'] == True) & (models['reg'] == 'layer'))
            | ((models['prox'] == False) & (models['reg'] == 'path'))]

    # create error and robust error columns
    err_piv, rob_err = pivot_error(errors, error=error)

    # join the model and errors table, compute mean
    model_errors = models.join(err_piv)
    model_errors = model_errors.groupby(
            ['lr', 'lambda_', 'dataset', 'prox', 'reg'])
    model_errors = model_errors.agg({
        'error': 'mean', 'nnz': 'mean', rob_err: 'mean'})
    model_errors.reset_index(level=model_errors.index.names, inplace=True)

    # remove entries with high error rates
    model_errors = model_errors[model_errors['error'] < 80.]
    model_errors = model_errors[model_errors[rob_err] < 90.]

    # select learning rate which achieves best error and robust error mean
    model_errors['score'] = (
            model_errors['error'] + model_errors[rob_err])
    model_errors['best score'] = model_errors.groupby(
            ['lambda_', 'dataset', 'prox', 'reg'])['score'].transform('min')
    model_errors = (
            model_errors[model_errors['score'] == model_errors['best score']])
    model_errors.sort_values(by='lambda_', inplace=True)

    error_vs_robust_plot = sns.relplot(
            x='error', y=rob_err, hue='reg', style='reg', markers=True,
            data=model_errors, edgecolor=None,
            alpha=0.5, row='dataset', kind='scatter')

    error_vs_robust_plot.savefig('error_vs_robust.pdf')


def error_vs_lambda_plot(
        models, errors, epochs=20, batch_size=100, hidden=200, error=0.1):
    """
    Maybe remove subgradient plots and add layer-wise constraint trained
    networks to the plot? We would have to invert the lambda_ for those

    """
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)
    models = models[models['reg'] != 'layer']
    models = models[models['lambda_'] < 10]
    err_piv, rob_err = pivot_error(errors, error=error)
    model_errors = models.join(err_piv)
    model_errors = model_errors[model_errors['error'] < 80.]
    model_errors = model_errors[model_errors[rob_err] < 90.]
    model_errors[r'$\lambda$'] = model_errors['lambda_']

    error_vs_lambda_plot = sns.relplot(
            x=r'$\lambda$', y='error', style='prox', hue='reg', markers=True,
            data=model_errors,
            alpha=1., col='dataset', kind='line')
    for ax in error_vs_lambda_plot.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
    error_vs_lambda_plot.savefig('error_vs_lambda.pdf')

    robust_vs_lambda_plot = sns.relplot(
            x=r'$\lambda$', y=rob_err, style='prox', hue='reg', markers=True,
            data=model_errors,
            alpha=1., col='dataset', kind='line')
    for ax in robust_vs_lambda_plot.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
    robust_vs_lambda_plot.savefig('robust_vs_lambda.pdf')


def iterplots(models, iters, epochs=20, batch_size=100, hidden=200):
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)
    models.drop(['nnz'], axis='columns', inplace=True)
    # pdb.set_trace()
    models = models[models['lr'] == 0.0001]
    models = models[
            (((models['reg'] == r'$\ell_1$') & (models['lambda_'] == 5e-1)) |
             ((models['reg'] == 'path') & (models['lambda_'] == 0.5)))]
    models = models[(models['reg'] == r'$\ell_1$') | (models['reg'] == 'path')]
    models = models[(models['dataset'] == 'fmnist') | (models['dataset'] == 'kmnist')]

    model_iters = models.join(iters)
    model_iters['iter'] = model_iters['iter'] + 1
    model_iters['iteration'] = model_iters['iter']
    model_iters['dataset/reg'] = model_iters[['dataset', 'reg']].agg('/'.join, axis=1)

    figsize = (15, 1)
    height = 2.
    aspect = 1.2

    sns.set(rc={"font.size": 8,
                'figure.figsize': figsize,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
                "xtick.labelsize" : 7,
                "ytick.labelsize": 7,
                "figure.subplot.wspace": 0.0,
                "legend.fontsize": 8
               })
# for sparsity lamb=0.00010  lr0.5
    nnz_vs_iter_plot = sns.relplot(
            x='iteration', y='nnz', style='prox', markers=False,
            data=model_iters, ci=None,
            alpha=1., col='dataset/reg', kind='line', height=height, aspect=aspect)
    for ax in nnz_vs_iter_plot.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
    nnz_vs_iter_plot.savefig('nnz_vs_iter.pdf')

    plt.figure(figsize=figsize)
    loss_vs_iter_plot = sns.relplot(
            x='iteration', y='loss', style='prox', markers=False,
            data=model_iters, ci=None,
            alpha=1., col='dataset/reg', kind='line', height=height, aspect=aspect,
            facet_kws={'sharey': True, 'sharex': True})
    for ax in loss_vs_iter_plot.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
            ax_.set_yscale('log')
    loss_vs_iter_plot.savefig('loss_vs_iter.pdf')


def sparsity_plot(models, iters, epochs=20, batch_size=100, hidden=200):
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)
    models.drop(['nnz'], axis='columns', inplace=True)
    models = models[((models['lr'] == 0.5) & (models['reg'] == 'path')) | ((models['lr'] == 1e-3) & (models['reg'] == r'$\ell_1$'))]
    models = models[
            (((models['reg'] == r'$\ell_1$') & (models['lambda_'] == 1e-2)) |
             ((models['reg'] == 'path') & (models['lambda_'] == 0.0001)))]
    models = models[(models['reg'] == r'$\ell_1$') | (models['reg'] == 'path')]
    models = models[(models['dataset'] == 'fmnist') | (models['dataset'] == 'mnist')]

    model_iters = models.join(iters)
    model_iters['iter'] = model_iters['iter'] + 1
    model_iters['iteration'] = model_iters['iter']
    model_iters['dataset/reg'] = model_iters[['dataset', 'reg']].agg('/'.join, axis=1)

    figsize = (15, 1)
    height = 2.
    aspect = 1.2

    sns.set(rc={"font.size": 8,
                'figure.figsize': figsize,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
                "xtick.labelsize" : 7,
                "ytick.labelsize": 7,
                "figure.subplot.wspace": 0.0,
                "legend.fontsize": 8
               })

    # for sparsity lamb=0.00010  lr0.5
    nnz_vs_iter_plot = sns.relplot(
            x='iteration', y='nnz', style='prox', markers=False,
            data=model_iters, ci=None,
            alpha=1., col='dataset/reg', kind='line', height=height, aspect=aspect)
    for ax in nnz_vs_iter_plot.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
    nnz_vs_iter_plot.savefig('sparisty_vs_iter.pdf')



def mnist_lambda_lr_path_nnz(models, iters, epochs=20, batch_size=100, hidden=200):
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)
    models = models[models['dataset'] == 'fmnist']
    models = models[models['prox'] == True]
    models = models[models['reg'] == 'path']
    # models = models[models['lr'] == .5]
    res = sns.relplot(y='nnz', x='lambda_', row='lr', data=models)
    for ax in res.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
    res.savefig('nnz_vs_lr_lambda.pdf')


def loss_vs_lr(
        models, errors, epochs=20, batch_size=100, hidden=200, error=0.1):
    models = subset_models(
            models, epochs=epochs, batch_size=batch_size, hidden=hidden)

    err_piv, rob_err = pivot_error(errors, error=error)

    # join the model and errors table, compute mean
    model_errors = models.join(err_piv)
    model_errors_l1 = model_errors[model_errors['reg'] == r'$\ell_1$']

    loss_vs_lr_plot_l1 = sns.relplot(
            x='lambda_', y='error', style='prox', row='lr', markers=False,
            data=model_errors_l1, ci=None,
            alpha=1., col='dataset', kind='line',
            facet_kws={'sharey': False, 'sharex': True})

    for ax in loss_vs_lr_plot_l1.axes:
        for ax_ in ax:
            ax_.set_xscale('log')
            # ax_.set_yscale('log')

    loss_vs_lr_plot_l1.savefig('loss_vs_lr_plot_l1.pdf')


def main(**kwargs):
    # pdb.set_trace()
    sns.set()
    sns.set(rc={"font.size": 8,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
                "xtick.labelsize" : 7,
                "ytick.labelsize": 7,
                "figure.subplot.wspace": 0.0,
                "legend.fontsize": 8
               })
    models = pd.read_pickle(MODEL_DATA_FILE)
    models.loc[models['reg'] == 'l1', 'reg'] = r'$\ell_1$'
    models.loc[models['reg'] == 'linfpath', 'reg'] = 'path'
    models.loc[models['reg'] == 'linfproj', 'reg'] = 'layer'
    models.loc[models['reg'] == 'layer', 'lambda_'] = (
            models.loc[models['reg'] == 'layer', 'lambda_'].apply(np.reciprocal))

#     errors = pd.read_pickle(TEST_ERRORS_DATA_FILE).set_index('file')
    iters = pd.read_pickle(ITER_DATA_FILE).set_index('file')

#     robust_vs_error_plot(
#             models.copy(deep=True), errors.copy(deep=True), error=0.1)
#     error_vs_lambda_plot(
#             models.copy(deep=True), errors.copy(deep=True), error=0.1)
    iterplots(
            models.copy(deep=True), iters.copy(deep=True))
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

