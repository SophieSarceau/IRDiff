#!/bin/bash
#SBATCH --job-name=irdiff-eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zxz.krypton@outlook.com
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH -o /blue/yanjun.li/pfq7pm.virginia/AIDD/IRDiff/slurm_out/irdiff-eval.out
#SBATCH --time=72:00:00

python eval_split.py --eval_start_index 0 --eval_end_index 99 \
                     --sample_path ./sampled_results/n128k3s5 \
                     --result_path ./eval_results/n128k3s5
