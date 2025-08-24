#!/bin/bash
#SBATCH --job-name=ave_wino
#SBATCH --output=%j_nsys.out
#SBATCH --error=%j_nsys.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --partition=V100
#SBATCH --gpus=1

LAYER_ID=$1
KERNEL=$2

SKIP=$((LAYER_ID * 4 + 1))
COUNT=3

source /pxe/opt/spack/share/spack/setup-env.sh
spack env activate hpc101-cuda

export TMPDIR=~/tmp
mkdir -p $TMPDIR

/usr/local/cuda/bin/nsys profile --output nsys_prof --stats=true \
                         -- ./winograd inputs/config.txt
