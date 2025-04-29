#!/bin/bash
###SBATCH -p HENON
#SBATCH --nodes 1
#SBATCH --cpus-per-task 14
#SBATCH --time 48:00:00
#SBATCH --job-name sim-H
#SBATCH -o /hildafs/projects/phy200017p/berlfein/jupyter_log/jupyter-notebook-%J.log
#SBATCH -e /hildafs/projects/phy200017p/berlfein/jupyter_log/jupyter-notebook-%J.log

cd $SLURM_SUBMIT_DIR
python run_sims_chromatic_final.py --filter_name H158 --shear_value 0.00 --start_idx 0  --end_index 10000  --saveInfo True --drawGal True --drawPSF True