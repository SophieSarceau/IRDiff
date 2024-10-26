#!/bin/bash
#SBATCH --job-name=irdiff-train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zxz.krypton@outlook.com
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:2
#SBATCH -o /blue/yanjun.li/pfq7pm.virginia/AIDD/IRDiff/slurm_out/irdiff-train-pl.out
#SBATCH --time=144:00:00

srun python train_pl.py
