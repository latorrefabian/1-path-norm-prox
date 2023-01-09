# import pdb
import argparse
import defaults
import os
import pandas as pd
import pickle

from defaults import CV_MODEL_DATA_FILE as MODEL_DATA_FILE
from defaults import CV_ERRORS_DATAFILE as TEST_ERRORS_DATA_FILE
from defaults import help_
from tqdm import tqdm

from rsparse.utils import filename_to_params
SAVE_EVERY = 10000


class DataBaseError(Exception):
    """
    Define exceptions when reading and writing to the serialized
    pandas.DataFrame objects, containing the data from the experiments.

    """
    pass


def write_model_data(ckpt_dir, df_file, delete_missing=False):
    """
    Takes all trained models in a directory, and appends the information about
    its train loop parameters, as well as properties of the final weights like
    percentage of nonzero weights. The dataframe created is indexed by
    filename.

    Args:
        ckpt_dir (str): path to the folder where the PyTorch weight files (.pt)
            are located.
        df_file (str): path to the pickle (.pkl) file where the data will be
            written. The file will be created if it does not exist. Note that
            if a previous record with the same parameters is found, it will
            not be duplicated.
        delete_missing (bool): TODO should look for entries in the dataframe
            for which the corresponding file no longer exists, and delete the
            entry.

    Raises:
        DataBaseError: If the name of columns of the pandas.DataFrame that
        exists in memory deviates from the entries that are trying to be added.

    """
    f = os.listdir(ckpt_dir)[0]
    f_wo_ext = os.path.splitext(f)[0]  # file without extension (e.g., '.pt')
    params = filename_to_params(f_wo_ext)  # remove .pt

    if os.path.isfile(df_file):
        df = pd.read_pickle(df_file)
        if not set(params.keys()) == set(df.columns):
            raise DataBaseError('Schema has changed. Reset the database.')
    else:
        params['file'] = f_wo_ext
        df = pd.DataFrame(columns=[k for k in params.keys()])
        df = df.set_index('file')

    if delete_missing:
        for f in df.index:
            if not os.path.isfile(os.path.join(ckpt_dir, f + '.pt')):
                os.remove(f)

    entries = list()

    files_ = os.listdir(ckpt_dir)
    remaining_files = list()
    for f in files_:
        f_wo_ext = os.path.splitext(f)[0]  # file without extension
        if f_wo_ext in df.index:
            pass
        else:
            remaining_files.append(f)

    for f in tqdm(remaining_files):
        f_wo_ext = os.path.splitext(f)[0]  # file without extension
        if f_wo_ext in df.index:
            continue
        params = filename_to_params(f_wo_ext)  # remove .pt
        params['file'] = f_wo_ext
        entries.append(params)
        if len(entries) > SAVE_EVERY:
            new_rows = pd.DataFrame(entries)
            new_rows = new_rows.set_index('file')
            df = pd.concat([df, new_rows], sort=True)
            df.to_pickle(df_file)
            entries = list()

    if len(entries) > 0:
        new_rows = pd.DataFrame(entries)
        new_rows = new_rows.set_index('file')
        df = pd.concat([df, new_rows], sort=True)
        df.to_pickle(df_file)


def update_test_errors(errors_dir, df_file):
    """
    Takes all trained models in a directory, and appends the information about
    their test error on adversarial examples, for a range of perturbation sizes
    specified in the 'epsilons' list defined in the update_dataframes.py
    script. The dataframe created is indexed by sequentially increasing
    integers, and its schema as of 01.27.2020 is the following:

    1. 'file' (str): name of the model's weight file.
    2. 'eps' (float): perturbation size
    3. 'test_error' (float): percentage of error in the perturbed test set.

    Args:
        ckpt_dir (str): path to the folder where the PyTorch weight files (.pt)
            are located.
        df_file (str): path to the pickle (.pkl) file where the data will be
            written. The file will be created if it does not exist. Note that
            if a previous record with the same parameters is found, it will
            not be duplicated.
        delete_missing (bool): TODO should look for entries in the dataframe
            for which the corresponding file no longer exists, and delete the
            entry.

    Raises:
        DataBaseError: If the name of columns of the pandas.DataFrame that
        exists in memory deviates from the entries that are trying to be added.

    """
    entries = list()
    columns = ('file', 'eps', 'test_error')

    if os.path.isfile(df_file):
        df = pd.read_pickle(df_file)
        if not set(columns) == set(df.columns):
            raise DataBaseError('Schema has changed. Reset the database.')
        existing = [(row['file'], row['eps']) for _, row in df.iterrows()]
        existing = set(existing)
    else:
        df = pd.DataFrame(columns=columns)
        existing = set()

    entries = list()

    for f in tqdm(os.listdir(errors_dir)):
        with open(os.path.join(errors_dir, f), 'rb') as handle:
            vals = pickle.load(handle)

        for val in vals:
            if (val['file'], val['eps']) in existing:
                continue
            else:
                entries.append(val)

    if len(entries) > 0:
        new_rows = pd.DataFrame(entries)
        df = pd.concat([df, new_rows], sort=True)
        df.to_pickle(df_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--ckpt_dir', type=str, default=defaults.CV_CKPTS_DIR,
            help=help_['cv_ckpt_dir'])
    parser.add_argument(
            '--errors_dir', type=str, default=defaults.CV_ERRORS_DIR,
            help=help_['errors_dir'])
    args = parser.parse_args()

    print('Writing model data...')
    write_model_data(
            ckpt_dir=args.ckpt_dir, df_file=MODEL_DATA_FILE)

    print('Updating test errors...')
    update_test_errors(
            errors_dir=args.errors_dir, df_file=TEST_ERRORS_DATA_FILE)

