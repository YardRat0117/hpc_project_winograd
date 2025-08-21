#!/bin/bash

source /pxe/opt/spack/share/spack/setup-env.sh
spack env activate hpc101-cuda

make
