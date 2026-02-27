# Issue #96: Race Condition in Pipelined Matmul

## Summary

When `T.Pipelined` is used with `num_stages > 0` in a matmul kernel, the
generated CUDA code contains a race condition that produces wrong results on
RTX 4070 Ti and A100 GPUs for large matrix sizes (e.g., N=8192).

## Bug Description

The `T.Pipelined` loop with `num_stages=3` generates a software-pipelined
kernel that prefetches the *next* tile of A and B into shared memory while
computing the *current* tile. The issue is that the synchronization between
pipeline stages is incorrect:

Looking at the generated CUDA kernel, the main loop body is:
```
for (int ko = 0; ko < 254; ++ko) {
    __syncthreads();                  // <-- sync BEFORE writing new data
    ... write stage (ko+2)%3 ...      // prefetch next-next tile
    __syncthreads();                  // <-- sync AFTER writing
    ... mma using stage (ko%3) ...    // compute current tile
}
```

The problem: after the first `__syncthreads()`, threads immediately start
writing into the `(ko+2) % 3` shared memory slot. But this slot may still be
in active use by MMA instructions from the *previous* loop iteration that
haven't fully retired yet. There is no guarantee that the MMA consuming slot
`(ko+2) % 3` from two iterations ago has completed before threads overwrite it.

This is a classic **write-after-read (WAR) hazard** in a triple-buffered
pipeline — the producer overwrites a buffer before the consumer is done with it.

## Root Cause

The `T.Pipelined` abstraction in TileLang does not correctly enforce the
read-before-write ordering across pipeline stage boundaries. The generated
`__syncthreads()` placement is insufficient to guarantee that a shared memory
slot has been fully consumed before being reused.

## Reproducing
```bash
python reproduce.py
```

The script compiles the buggy kernel and prints the generated CUDA source.
The race is visible in the loop structure: two `__syncthreads()` barriers
frame the *write* phase, but there is no barrier separating the *read* (MMA)
phase of the previous iteration from the write phase of the current one.

**Note:** Actually triggering wrong output at runtime requires:
- An SM80+ GPU (A100, H100) — requires `CUTE_ARCH_MMA_SM80_ENABLED`
- N=8192 (large enough that the race window is hit reliably)
- The MX450 (SM75) cannot run this kernel at all; it crashes with
  `SM80_16x8x16_F32F16F16F32_TN without CUTE_ARCH_MMA_SM80_ENABLED`

## The Fix

Setting `num_stages=0` disables the pipeline entirely, falling back to a
simple synchronous load-compute pattern that is always correct:
```python
# Buggy
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):

# Fixed
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
```

## Environment

- TileLang version: 0.1.8
- Python: 3.10
- Test GPU: NVIDIA GeForce MX450 (SM75) — kernel generation only
- Affected GPUs: RTX 4070 Ti, A100 (confirmed wrong results in issue report)
