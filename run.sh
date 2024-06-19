#!/bin/bash -lT
#SBATCH -J OCTA
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -p gpu_7day
#SBATCH --gpus 4
#SBATCH --mem-per-gpu 24G
#SBATCH -w dscog011
#SBATCH -t 48:00:00

conda activate octa

nvidia-smi

python eval.py