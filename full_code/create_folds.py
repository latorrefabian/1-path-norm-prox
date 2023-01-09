import pickle
import os
import numpy as np

from defaults import FOLDS_DIR
from rsparse.utils import fold_filename, get_data
from sklearn.model_selection import KFold


def save_folds(folder, n_splits, dataset, dataset_name):
    """
    Write to disk indices of the folds for a given dataset
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


if __name__ == '__main__':
    n_splits = 4
    for name in ['mnist', 'fmnist', 'kmnist', 'cifar10']:
        train, test = get_data(name)
        save_folds(
                FOLDS_DIR, n_splits=n_splits, dataset=train, dataset_name=name)

