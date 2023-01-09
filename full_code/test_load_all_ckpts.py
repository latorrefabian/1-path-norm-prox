import pdb
import argparse
import torch
import os
import time

from rsparse.utils import filename_to_params, get_input_output_size
from rsparse.lipschitz import OneLayerNetwork, ELU
from tqdm import tqdm
import pickle

CKPT_DIR = '/scratch/latorre/robust_sparsity/ckpts'

OK_FILES = 'ok_files.pkl'


def main(f, ok_files):
    f_wo_ext = os.path.splitext(f)[0]  # file without extension
    params = filename_to_params(f_wo_ext)  # remove .pt
    weights_file = os.path.join(CKPT_DIR, f)
    in_features, out_features = get_input_output_size(params['dataset'])
    network = OneLayerNetwork(
            in_features=in_features, hidden_features=params['hidden'],
            out_features=out_features, activation=ELU())
    try:
        network.load_state_dict(torch.load(weights_file))
        ok_files.add(f)
        with open(OK_FILES, 'wb') as handle:
            pickle.dump(ok_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except RuntimeError as e:
        os.remove(weights_file)
        print('deleted corrupt file ' + str(f))
        return



if __name__ == '__main__':
    if not os.path.isfile(OK_FILES):
        ok_files = set()
    else:
        with open(OK_FILES, 'rb') as handle:
            ok_files = pickle.load(handle)
    print('number of ok files ' + str(len(ok_files)))
    time.sleep(3)
    for f in sorted(os.listdir(CKPT_DIR)):
        if f in ok_files:
            continue
        print(f)
        main(f, ok_files)
    
