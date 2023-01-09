import pdb
import argparse
import pandas
import os
import subprocess


from defaults import INTERPRETER
from defaults import CV_MODEL_DATA_FILE, CV_ERRORS_DATA_FILE
from defaults import LOG_DIR
from defaults import FINAL_CKPT_DIR
from rsparse.utils import filename_to_params

from defaults import template_str_helvetios as template_str
from jinja2 import Template
from rsparse.utils import params_to_filename, params_dict_to_str

scripts_per_job = 6
pathprox_scripts_per_job = 1
CPUS = 18
TIME = {
    'linfpath-prox': '2:00:00',
    'other': '2:00:00',
}


def choose_best_params(models, errors):
    err_piv, rob_err = pivot_error(errors, error=0.1)
    model_errors = models.join(err_piv)
    model_errors['hidden'] = model_errors['hidden'].map(str)
    columns = set(model_errors.columns)
    columns.remove('fold')
    columns.remove('error')
    columns.remove('robust error')
    model_errors['count'] = (
            model_errors
            .groupby(list(columns))['fold'].transform(len))
    model_errors = model_errors[
            model_errors['n_folds'] == model_errors['count']]
    model_errors = model_errors.groupby(list(columns))
    model_errors = model_errors.agg({
        'error': 'mean', 'robust error': 'mean'})
    model_errors.reset_index(level=model_errors.index.names, inplace=True)
    model_errors['score'] = (
            model_errors['robust error'])
    model_errors['best'] = (
            model_errors
            .groupby(['dataset', 'reg'])['score'].transform('min'))
    best_models = model_errors[model_errors['score'] == model_errors['best']]
    pdb.set_trace()
    return best_models.index.map(lambda x: filename_to_params(x))


def train_models(params, script_dir, overwrite=False):
    script = os.path.join(script_dir, 'profile.py')
    template = Template(template_str)
    ckpt_dir = os.path.join(script_dir, FINAL_CKPT_DIR)
    log_dir = os.path.join(script_dir, LOG_DIR)

    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    param_list = list()
    param_list_pathprox = list()

    for i, param in enumerate(params):
        filename_ = params_to_filename(**param)
        ckpt = os.path.join(ckpt_dir, filename_ + '.pt')

        if os.path.isfile(ckpt) and not overwrite:
            continue

        param['ckpt_dir'] = ckpt_dir
        param['overwrite'] = overwrite
        param_str = params_dict_to_str(param)

        if param['reg'] == 'linfpath' and param['prox']:
            param_list_pathprox.append(param_str)
            if len(param_list_pathprox) > pathprox_scripts_per_job:
                with open('job_script.sh', 'w') as f:
                    f.write(template.render(
                        param_list=param_list_pathprox, script=script,
                        log_dir=log_dir, cpus_per_task=CPUS,
                        time=TIME['linfpath-prox'], **args))
                subprocess.run('sbatch job_script.sh', shell=True)
                param_list_pathprox = list()

        param_list.append(param_str)

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


def main(script_dir, epochs, **args):
    models = pandas.read_pickle(CV_MODEL_DATA_FILE)
    errors = pandas.read_pickle(CV_ERRORS_DATA_FILE).set_index('file')
    best_params = choose_best_params(models, errors)
    for param in best_params:
        param['epochs'] = epochs
    train_models(best_params, script_dir, epochs, **args)
    train_models(best_params, script_dir, epochs)


def pivot_error(errors, error=0.1):
    err_piv = errors.pivot(columns='eps')
    err_piv['error'] = err_piv['test_error'][0.0]
    rob_err = 'robust error'
    err_piv[rob_err] = err_piv['test_error'][error]
    err_piv = err_piv[['error', rob_err]]
    err_piv.columns = err_piv.columns.get_level_values(0)
    return err_piv, rob_err


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--interpreter', type=str,
            default=INTERPRETER['helvetios'])
    parser.add_argument(
            '--script_dir', type=str,
            default='/scratch/latorre/robust_sparsity')

    # SLURM parameters
    parser.add_argument('--job_name', type=str, default='final_robsparse')
    parser.add_argument('--mem', type=str, default='4GB')
    parser.add_argument('--nodes', type=str, default='1')
    parser.add_argument('--overwrite', action='store_true')

    # other parameters
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))

