1. For compiling, please run the `compile.sh`.
2. For running, please submit the `run.sh` to `slurm` via the `sbatch` command.
3. For clearing all `*.out`, `*.err` and compiled binary file `winograd`, please run the `clear.sh`.
4. For running the Nsight Compute profiling, please submit the `ncu.sh` to `slurm` via the `sbatch` command, with the format: `sbatch ncu.sh <layer_id> <mode>`. `<layer_id>` corresponds to the input case layer, while `<mode>` corresponds to the implementation (naive/winograd).
