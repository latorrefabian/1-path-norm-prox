# import pdb
import argparse
import os
import subprocess

from defaults import template_str_profile as template_str
from jinja2 import Template
from itertools import product
from rsparse.utils import params_to_filename, params_dict_to_str

from defaults import var_dict, CKPT_DIR, DATA_DIR, LOG_DIR

# all possible combinations for the python script's parameters

var_lists = list()

for k in var_dict.keys():
    var_lists.append([(k, v) for v in var_dict[k]])


def main(vars_, script_dir, overwrite, **args):
    script = os.path.join(script_dir, 'profile.py')
    template = Template(template_str)
    ckpt_dir = os.path.join(script_dir, CKPT_DIR)
    data_dir = os.path.join(script_dir, DATA_DIR)
    log_dir = os.path.join(script_dir, LOG_DIR)

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    for i, x in enumerate(product(*var_lists)):
        x = dict(x)

        # exclude product + proximal
        if x['prox'] and x['reg'] == 'linfprod':
            continue
        if x['prox'] and x['reg'] == 'linfpath':
            args['time'] = '7:00:00'
        else:
            args['time'] = '1:00:00'

        if x['reg'] == 'linfproj':
            if x['prox'] and x['lambda_'] < 1e-2:
                continue
        else:
            if x['lambda_'] > 1e2:
                continue

        filename_ = params_to_filename(**x)
        ckpt = os.path.join(ckpt_dir, filename_ + '.pt')

        if os.path.isfile(ckpt) and not overwrite:
            continue

        x['ckpt_dir'] = ckpt_dir
        x['data_dir'] = data_dir
        x['overwrite'] = overwrite
        params = params_dict_to_str(x)

        with open('job_script.sh', 'w') as f:
            f.write(template.render(
                params=params, script=script, log_dir=log_dir, **args))

        subprocess.run('sbatch job_script.sh', shell=True)

    if os.path.isfile('job_script.sh'):
        os.remove('job_script.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # python interpreter and where the project folder is located
    parser.add_argument(
            '--interpreter', type=str,
            default='/home/latorre/miniconda3/envs/pytorch/bin/python')
    parser.add_argument(
            '--script_dir', type=str,
            default='/home/latorre/robust_sparsity')

    # SLURM parameters
    parser.add_argument('--job_name', type=str, default='robsparse')
    parser.add_argument('--time', type=str, default='2:30:00')
    parser.add_argument('--mem', type=str, default='4GB')
    # parser.add_argument('--partition', type=str, default='gpu')
    parser.add_argument('--nodes', type=str, default='1')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(vars_=var_dict, **vars(args))

