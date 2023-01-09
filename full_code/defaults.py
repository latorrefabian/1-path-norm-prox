MODEL_DATA_FILE = 'models.pkl'
TEST_ERRORS_DATA_FILE = 'test_errors.pkl'
ITER_DATA_FILE = 'iter_data.pkl'

CKPT_DIR = 'ckpts'
DATA_DIR = 'data'
ERRORS_DIR = 'errors'
LOG_DIR = 'log'

NEW_MODEL_DATA_FILE = 'new_models.pkl'
NEW_TEST_ERRORS_DATA_FILE = 'new_test_errors.pkl'
NEW_ITER_DATA_FILE = 'new_iter_data.pkl'

NEW_CKPT_DIR = 'new_ckpts'
NEW_DATA_DIR = 'new_data'
NEW_ERRORS_DIR = 'new_errors'

CV_MODEL_DATA_FILE = 'cv_models.pkl'
CV_ERRORS_DATA_FILE = 'cv_test_errors.pkl'

CV_CKPT_DIR = 'cv_ckpts'
CV_ERRORS_DIR = 'cv_errors'
FOLDS_DIR = 'cv_folds'

FINAL_CKPT_DIR = 'final_ckpts'
FINAL_ERRORS_DIR = 'final_errors'

INTERPRETER = {
    'simba': '/home/latorre/miniconda3/envs/pytorch/bin/python',
    'helvetios': '/home/latorre/.virtualenvs/test/bin/python',
}

# epsilons = {
#     'mnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'fmnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'kmnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'emnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     # 'mnist': [0.05]
# }

epsilons = {
    'mnist': [0., 0.1, ],  # 0.15, 0.2, 0.25, 0.3],
    'fmnist': [0., 0.1, ],  # 0.15, 0.2, 0.25, 0.3],
    'kmnist': [0., 0.1, ],  # 0.15, 0.2, 0.25, 0.3],
}

# epsilons = {
#     'mnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'fmnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'kmnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     'emnist': [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
#     # 'mnist': [0.05]
# }

var_dict = {
    'seed': [0, 1, 2, 3, 4, 5, ],
    'batch_size': [100],
    'reg': ['l1', 'linfproj', 'linfpath'],
    'epochs': [20],
    'lr': [1e-1, 1e-2, 1e-3, 1e-4, 5e-1, 5e-2, 5e-3, 5e-4],
    'prox': [True, False],
    'dataset': ['mnist', 'fmnist', 'kmnist'],
    'hidden': [200, ],
    'cuda': [False],
    'lambda_': [0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] +
               [3e-5, 3e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1, ] +
               [5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e0, 5e1, ],
    # 'lambda_': [0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] +
    #            [2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2] +
    #            [3e-5, 3e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1, 3e2] +
    #            [4e-5, 4e-4, 4e-3, 4e-2, 4e-1, 4e0, 4e1, 4e2] +
    #            [5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e0, 5e1, 5e2],
}

var_dict_cv = {
    'seed': [0],
    'batch_size': [100],
    'reg': ['l1', 'linfproj', 'linfpath'],
    'epochs': [10],
    'lr': [1e-1, 1e-2, 1e-3, 5e-1, 5e-2, 5e-3],
    'prox': [True],
    'dataset': ['mnist', 'fmnist', 'kmnist'],
    # 'hidden': [[100], [100, 100, 100]],
    'hidden': [[100, 100, 100]],
    'cuda': [False],
    'lambda_': [0., 1e-4, 1e-3, 1e-2, 1e-1, 1e0, ] +
               [3e-4, 3e-3, 3e-2, 3e-1, 3e0, ] +
               [5e-4, 5e-3, 5e-2, 5e-1, ],
    'n_folds': [4],
    'fold': [0, 1, 2, 3],
}

new_var_dict = {
    'seed': [0, 1, ],
    'batch_size': [100],
    # 'reg': ['l1', 'linfproj', 'linfpath', 'linfprod'],
    'reg': ['l1', 'linfpath'],
    'epochs': [10],
    'lr': [1e-1, 1e-2, 1e-3, 1e-4, 5e-1, 5e-2, 5e-3, 5e-4],
    'prox': [True, False],
    # 'dataset': ['mnist', 'fmnist', 'kmnist',], #'emnist'],
    'dataset': ['fmnist', 'kmnist', ],
    'hidden': [200, ],
    'cuda': [False],
    'lambda_': [0., 1e-4, 1e-3, 1e-2, 1e-1, 1e0, ] +
               [3e-4, 3e-3, 3e-2, 3e-1, 3e0, ] +
               [5e-4, 5e-3, 5e-2, 5e-1, ],
    # 'lambda_': [0., 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2] +
    #            [2e-5, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 2e1, 2e2] +
    #            [3e-5, 3e-4, 3e-3, 3e-2, 3e-1, 3e0, 3e1, 3e2] +
    #            [4e-5, 4e-4, 4e-3, 4e-2, 4e-1, 4e0, 4e1, 4e2] +
    #            [5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 5e0, 5e1, 5e2],
}


help_ = {
    'seed': 'random seed for random number generator',
    'batch_size': 'batch size for stochastic gradient computation',
    'reg': 'regularizer to use, one of "l1", "linfproj", '
           '"linfpath"',
    'epochs': 'number of epochs to train for (passes over dataset)',
    'lr': 'learning rate for optimization algorithm',
    'prox': 'if this flag is added, proximal stochastic gradient descent '
            'will be used for optimization',
    'dataset': 'name of the dataset, one of "mnist"',
    'hidden': 'list of number of hidden neurons in each hidden layer of the '
              'network',
    'n_folds': 'number of folds for cross-validation',
    'fold': 'index of fold to train the model',
    'ckpt_dir': 'directory where to store trained networks',
    'data_dir': 'directory where to store '
                'information about the training loop',
    'errors_dir': 'directory where to store '
                  'information about test error on adversarial examples',
    'folds_dir': 'directory where fold indices are stored',
    'cuda': 'if this flag is added, default gpu will be used for training',
    'lambda_': 'penalty parameter of the regularizer',
    'overwrite': 'if True, a model will be trained and saved even if '
                 'an older model with exactly the same configuration exists',
    'fgsm': 'bound on the L-infinity magnitude of the adversarial perturbation'
            'used for adversarial training. defaults to zero'
}


# add the following to receive emails
# #SBATCH --mail-user=fabian.latorre@epfl.ch
# #SBATCH --mail-type=ALL

template_str = """#!/bin/bash

#
#SBATCH --job-name={{ job_name }}
#SBATCH --time={{ time }}
#SBATCH --mem={{ mem }}
#SBATCH --nodes={{ nodes }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --output={{ log_dir }}/%j.out

PYTHON={{ interpreter }}
$PYTHON {{ script }} {{ params }}
"""
template_str_multi = """#!/bin/bash
#
#SBATCH --job-name={{ job_name }}
#SBATCH --time={{ time }}
#SBATCH --mem={{ mem }}
#SBATCH --nodes={{ nodes }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --output={{ log_dir }}/%j.out

PYTHON={{ interpreter }}
{% for params in param_list %}
$PYTHON {{ script }} {{ params }}
{% endfor %}
"""

template_str_helvetios = """#!/bin/bash

#
#SBATCH --job-name={{ job_name }}
#SBATCH --time={{ time }}
#SBATCH --mem={{ mem }}
#SBATCH --nodes={{ nodes }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --output={{ log_dir }}/%j.out

module load gcc/8.3.0 python/3.7.3
PYTHON={{ interpreter }}
{% for params in param_list %}
$PYTHON {{ script }} {{ params }}
{% endfor %}
"""

template_str_helvetios_single = """#!/bin/bash

#
#SBATCH --job-name={{ job_name }}
#SBATCH --time={{ time }}
#SBATCH --mem={{ mem }}
#SBATCH --nodes={{ nodes }}
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH --output={{ log_dir }}/%j.out

module load gcc/8.3.0 python/3.7.3
PYTHON={{ interpreter }}
$PYTHON {{ script }} {{ params }}
"""

