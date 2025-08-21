> Note: this `TODO.md` should only be preserved during the development. It shouldn't be preserved for final submission.

1. ~~Basic Scripts: to build, run and clear~~ DONE
2. ~~Profile Script: to profile with Nsight COmpute~~ DONE
3. Baseline Proflie: to profile typical cases for the baseline TODO
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
4. Optimization v1: this would be based on `3. Baseline Profile` TODO
