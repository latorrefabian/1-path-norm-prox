# import pdb
import argparse
import os
import subprocess

from defaults import template_str_helvetios as template_str
from jinja2 import Template
from itertools import product
from rsparse.utils import params_to_filename, params_dict_to_str

from defaults import var_dict_cv as var_dict
from defaults import INTERPRETER
from defaults import CV_CKPT_DIR as CKPT_DIR
from defaults import LOG_DIR

# all possible combinations for the python script's parameters

scripts_per_job = 600
pathprox_scripts_per_job = 6
CPUS = 18
TIME = {
    'linfpath-prox': '6:00:00',
    'other': '6:00:00',
}

var_lists = list()

for k in var_dict.keys():
    var_lists.append([(k, v) for v in var_dict[k]])


def main(vars_, script_dir, overwrite, **args):
    script = os.path.join(script_dir, 'profile_cv.py')
    template = Template(template_str)
    ckpt_dir = os.path.join(script_dir, CKPT_DIR)
    log_dir = os.path.join(script_dir, LOG_DIR)

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    param_list = list()
    param_list_pathprox = list()

    for i, x in enumerate(product(*var_lists)):
        x = dict(x)

        filename_ = params_to_filename(**x)
        ckpt = os.path.join(ckpt_dir, filename_ + '.pt')

        if os.path.isfile(ckpt) and not overwrite:
            continue

        x['ckpt_dir'] = ckpt_dir
        x['overwrite'] = overwrite
        params = params_dict_to_str(x)

        if x['reg'] == 'linfpath' and x['prox']:
            param_list_pathprox.append(params)
            if len(param_list_pathprox) > pathprox_scripts_per_job:
                with open('job_script.sh', 'w') as f:
                    f.write(template.render(
                        param_list=param_list_pathprox, script=script,
                        log_dir=log_dir, cpus_per_task=CPUS,
                        time=TIME['linfpath-prox'], **args))
                subprocess.run('sbatch job_script.sh', shell=True)
                param_list_pathprox = list()

        param_list.append(params)

        if len(param_list) > scripts_per_job:
            with open('job_script.sh', 'w') as f:
                f.write(template.render(
                    param_list=param_list, script=script, log_dir=log_dir,
                    cpus_per_task=CPUS, time=TIME['other'], **args))

            subprocess.run('sbatch job_script.sh', shell=True)
            param_list = list()

    if len(param_list) > 0:
        with open('job_script.sh', 'w') as f:
            f.write(template.render(
                param_list=param_list, script=script, log_dir=log_dir,
                cpus_per_task=CPUS, time=TIME['other'], **args))

        subprocess.run('sbatch job_script.sh', shell=True)

    if len(param_list_pathprox) > 0:
        with open('job_script.sh', 'w') as f:
            f.write(template.render(
                param_list=param_list_pathprox, script=script, log_dir=log_dir,
                cpus_per_task=CPUS, time=TIME['linfpath-prox'], **args))
        subprocess.run('sbatch job_script.sh', shell=True)

    if os.path.isfile('job_script.sh'):
        os.remove('job_script.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # python interpreter and where the project folder is located
    parser.add_argument(
            '--interpreter', type=str,
            default=INTERPRETER['helvetios'])
    parser.add_argument(
            '--script_dir', type=str,
            default='/scratch/latorre/robust_sparsity')

    # SLURM parameters
    parser.add_argument('--job_name', type=str, default='cv_robsparse')
    # parser.add_argument('--time', type=str, default='4:00:00')
    parser.add_argument('--mem', type=str, default='4GB')
    # parser.add_argument('--partition', type=str, default='gpu')
    parser.add_argument('--nodes', type=str, default='1')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    main(vars_=var_dict, **vars(args))

