> Note: this `TODO.md` should only be preserved during the development. It shouldn't be preserved for final submission.

1. ~~Basic Scripts: to build, run and clear~~ DONE
2. ~~Profile Script: to profile with Nsight COmpute~~ DONE
3. ~~Baseline Proflie: to profile typical cases for the baseline~~ DONE
    1. Case 01
        - Layer 1 `32 112 112 64 64`
        - Big feature map, medium batch, small channel
        - Typical mem-bound
    2. Case 02
        - Layer 4 `128 112 112 128 64`
        - Big feature map, medium batch, medium channel
        - Balanced
    3. Case 03
        - Layer 9 `512 16 16 2048 64`
        - Small feature map,  big batch, big channel
        - Typical compute-bound
4. ~~Optimization v1~~ DONE
    1. Precompute and reuse filter transform
        - The `u_kc[4][4]` is computed in each channel of each thread.
        - We could try to store it (the `filter transform`) to shared memory.
    2. Loop unrolling with `#pragma unroll`
        - For fixed size matrix loops, we could try to use `#pragma unroll`.
        - Try this on `B_T * d`, `temp_d * B`, `A_T * m`, etc.
5. Profile v1-beta to profile typical cases for the baseline
    1. Test whether v1 runs correctly.
    2. (Critical) Fix the `ncu.sh` script for revised kernels.
    3. Run profiling on v1 to spot bottlenecks. Still layer 1, 4, and 9.
6. Further Optimization STAGED
    1. Use WMMA / Tensor Cores
        - Try use `wmma::load_matrix_sync`, `wmma::mma_sync`, `wmma::store_matrix_sync` to rewrite element-wise transform/accumulation
    2. Use MPI
