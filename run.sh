#!/bin/bash
#SBATCH --job-name=ave_wino
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:00:30
#SBATCH --partition=V100
#SBATCH --gpus=1

spack env activate hpc101-cuda

./winograd inputs/config.txt
