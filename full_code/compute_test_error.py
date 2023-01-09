import pdb
import argparse
import os
import pickle
import torch
import defaults
import multiprocessing


from tqdm import tqdm
from rsparse.lipschitz import OneLayerNetwork, ELU
from rsparse.utils import (
        filename_to_params, get_input_output_size, get_dataset,
        test_error)


def func(x):
    _main(*x)


def main(ckpt_dir, filename, errors_dir):
    filename = [(ckpt_dir, f, errors_dir) for f in filename]
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
        print('found slurm environment variable')
    except KeyError:
        ncpus = multiprocessing.cpu_count()
        print('did not find slurm environment variable')
    print('working with ' + str(ncpus) + ' cores')
    if ncpus == 1:
        print('sequential')
        for fil in tqdm(filename):
            func(fil)
    else:
        print('paralell')
        pool = multiprocessing.Pool()
        for _ in tqdm(pool.imap_unordered(func, filename), total=len(filename)):
            pass
        # pool.map(func, filename)
        pool.close()


def _main(ckpt_dir, filename, errors_dir):
    f_wo_ext = os.path.splitext(filename)[0]  # file without extension
    weights_file = os.path.join(ckpt_dir, filename)

    if not os.path.isfile(weights_file):
        raise ValueError('weight file does not exist')

    if not os.path.exists(errors_dir):
        os.mkdir(errors_dir)

    outfile = os.path.join(errors_dir, f_wo_ext + '.pkl')

    if os.path.isfile(outfile):
        with open(outfile, 'rb') as handle:
            entries = pickle.load(handle)
    else:
        entries = list()

    existing_eps = set([x['eps'] for x in entries])

    params = filename_to_params(f_wo_ext)  # remove .pt
    epsilons = defaults.epsilons[params['dataset']]
    in_features, out_features = get_input_output_size(params['dataset'])
    network = OneLayerNetwork(
            in_features=in_features, hidden_features=params['hidden'],
            out_features=out_features, activation=ELU())
    network.load_state_dict(torch.load(weights_file))

    test_loader = get_dataset(name=params['dataset'], batch_size=1)[1]

    for eps in epsilons:
        if eps in existing_eps:
            continue
        entry = {'file': f_wo_ext, 'eps': eps}
        entry['test_error'] = test_error(
                module=network, test_loader=test_loader, eps=eps)
        entries.append(entry)

    with open(outfile, 'wb') as handle:
        pickle.dump(entries, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--ckpt_dir', type=str, default=defaults.CKPT_DIR,
            help=defaults.help_['ckpt_dir'])
    parser.add_argument(
            '--errors_dir', type=str, default=defaults.ERRORS_DIR,
            help=defaults.help_['errors_dir'])
    parser.add_argument('--filename', type=str, nargs='+')
    args = parser.parse_args()
    main(**vars(args))

