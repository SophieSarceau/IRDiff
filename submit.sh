#!/bin/bash
#SBATCH --job-name=irdiff-sample
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zxz.krypton@outlook.com
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-cpu=5G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4
#SBATCH -o /blue/yanjun.li/pfq7pm.virginia/AIDD/IRDiff/slurm_out/irdiff-sample.out
#SBATCH --time=72:00:00

python sample_split.py --start_index 0 --end_index 99 --batch_size 25