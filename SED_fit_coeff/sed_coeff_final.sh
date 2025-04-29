#!/bin/bash
#SBATCH -p HENON
#SBATCH --nodes 1
#SBATCH --cpus-per-task 28
#SBATCH --time 48:00:00
#SBATCH --job-name SED_fit
#SBATCH -o /hildafs/projects/phy200017p/berlfein/jupyter_log/jupyter-notebook-%J.log
#SBATCH -e /hildafs/projects/phy200017p/berlfein/jupyter_log/jupyter-notebook-%J.log

cd $SLURM_SUBMIT_DIR
python sed_coeff_final.py