#!/bin/bash
#SBATCH --job-name=smatrix2    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -c 10
#SBATCH --gres=gpu:4
#SBATCH -A m1759
#SBATCH --ntasks-per-node=4
#SBATCH -q special
#SBATCH --output=serial_test2.log
pwd; hostname; date

module load pytorch/v1.5.1-gpu
#rm serial_test2.log
srun -N 1 python ~/admm/tests/nesap_hackathon/reconstruct_simul_data_multi_gpu_single_node.py

date