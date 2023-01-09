import pdb
import argparse
import os
import subprocess
import pickle

from defaults import epsilons
from defaults import template_str_helvetios_single as template_str
from jinja2 import Template
from rsparse.utils import params_dict_to_str, filename_to_params
from defaults import NEW_CKPT_DIR as CKPT_DIR
from defaults import NEW_ERRORS_DIR as ERRORS_DIR
from defaults import LOG_DIR, INTERPRETER


def main(script_dir, **args):
    pdb.set_trace()
    script = os.path.join(script_dir, 'compute_test_error.py')
    template = Template(template_str)

    ckpt_dir = os.path.join(script_dir, CKPT_DIR)
    errors_dir = os.path.join(script_dir, ERRORS_DIR)
    log_dir = os.path.join(script_dir, LOG_DIR)

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(errors_dir):
        os.mkdir(errors_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    filename = list()

    for f in os.listdir(ckpt_dir):
        f_wo_ext = os.path.splitext(f)[0]  # file without extension
        params = filename_to_params(f_wo_ext)  # remove .pt
        epsilons_ = epsilons[params['dataset']]

        outfile = os.path.join(errors_dir, f_wo_ext + '.pkl')

        if os.path.isfile(outfile):
            with open(outfile, 'rb') as handle:
                entries = pickle.load(handle)
            existing_eps = set([x['eps'] for x in entries])
            if all([eps in existing_eps for eps in epsilons_]):
                continue

        filename.append(f)

        if len(filename) < int(args['files_per_cpu']) * int(args['cpus_per_task']):
            continue

        params = {
                'ckpt_dir': ckpt_dir,
                'errors_dir': errors_dir,
                'filename': ' '.join(['"' + f + '"' for f in filename])
        }
        params = params_dict_to_str(params)

        with open('job_script.sh', 'w') as f:
            f.write(template.render(
                params=params, script=script, log_dir=log_dir, **args))
        subprocess.run('sbatch job_script.sh', shell=True)
        filename = list()

    if len(filename) > 0:
        params = {
                'ckpt_dir': ckpt_dir,
                'errors_dir': errors_dir,
                'filename': ' '.join(['"' + f + '"' for f in filename])
        }
        params = params_dict_to_str(params)

        with open('job_script.sh', 'w') as f:
            f.write(template.render(
                params=params, script=script, log_dir=log_dir, **args))

        subprocess.run('sbatch job_script.sh', shell=True)

    if os.path.isfile('job_script.sh'):
        os.remove('job_script.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    # python interpreter and script to run
    parser.add_argument(
            '--interpreter', type=str,
            default=INTERPRETER['helvetios'])
    parser.add_argument(
            '--script_dir', type=str,
            default='/scratch/latorre/robust_sparsity')

    # SLURM parameters
    parser.add_argument('--job_name', type=str, default='newtesterrors')
    parser.add_argument('--time', type=str, default='30:00')
    parser.add_argument('--mem', type=str, default='4GB')
    parser.add_argument('--cpus-per-task', type=str, default=1)
    parser.add_argument('--files-per-cpu', type=str, default=18)
    # parser.add_argument('--partition', type=str, default='gpu')
    parser.add_argument('--nodes', type=str, default='1')
    args = parser.parse_args()
    main(ckpt_dir=CKPT_DIR, errors_dir=ERRORS_DIR, **vars(args))

