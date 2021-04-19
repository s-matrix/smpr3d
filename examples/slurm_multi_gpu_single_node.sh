#!/bin/bash
#SBATCH --job-name=smatrix2    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 5
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH -A m1759
#SBATCH --ntasks-per-node=1
#SBATCH -q special
#SBATCH --output=serial_test2.log
pwd; hostname; date

module load cgpu
module load pytorch/1.8.0-gpu
#rm serial_test2.log
srun -N 1 python ~/smpr3d/examples/reconstruct_simul_data_multi_gpu_single_node.py

date