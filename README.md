1. For compiling, please run the `compile.sh`.
2. For running, please submit the `run.sh` to `slurm` via the `sbatch` command.
3. For clearing all `*.out`, `*.err` and compiled binary file `winograd`, please run the `clear.sh`.
4. For running the Nsight Compute profiling, please submit the `ncu.sh` to `slurm` via the `sbatch` command, with the format: `sbatch ncu.sh <layer_id> <kernel>`. `<layer_id>` corresponds to the input case layer, while `<kernel>` corresponds to the kernel implementation (kernel name).
5. For running the Nsight Systems profiling, please submit the `nsys.sh` to `slurm` via the `sbatch` command.
