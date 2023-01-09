import argparse
import torch
import os

from rsparse.utils import filename_to_params, get_input_output_size
from rsparse.lipschitz import OneLayerNetwork, ELU

CKPT_DIR = '/scratch/latorre/robust_sparsity/ckpts'


def main(f):
    f_wo_ext = os.path.splitext(f)[0]  # file without extension
    params = filename_to_params(f_wo_ext)  # remove .pt
    weights_file = os.path.join(CKPT_DIR, f)
    in_features, out_features = get_input_output_size(params['dataset'])
    network = OneLayerNetwork(
            in_features=in_features, hidden_features=params['hidden'],
            out_features=out_features, activation=ELU())
    network.load_state_dict(torch.load(weights_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()
    main(args.file)
    
