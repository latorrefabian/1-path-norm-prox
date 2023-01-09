#!/bin/bash
#
#SBATCH --job-name=updatedf
#SBATCH --time=2:00:00
#SBATCH --mem=24GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/latorre/robust_sparsity/log/updatedf_%j.out

module load gcc/8.3.0 python/3.7.3
PYTHON=/home/latorre/.virtualenvs/test/bin/python
$PYTHON /scratch/latorre/robust_sparsity/update_dataframes.py
