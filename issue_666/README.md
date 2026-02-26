# Issue #666 — Incorrect Results When Clearing Shared Memory Before Pipelined Loops

## Source
- **GitHub Issue:** https://github.com/tile-ai/tilelang/issues/666
- **Repo:** TileLang
- **Status:** Closed

## What is the Bug?

When `T.clear()` is called on a shared memory buffer BEFORE a pipelined loop
(`T.Pipelined` with `num_stages > 1`), TileLang generates CUDA code where the
clear operation races against the async pipeline stages on H100 GPUs.

The async pipeline prefetches data into shared memory buffers across multiple
stages. When a clear is inserted before the pipeline starts, the generated code
does not properly synchronize the clear with the async loads, causing threads
to read zeroed-out or stale memory instead of the actual data.

This results in 79.3% wrong output elements on H100. It works correctly on
RTX 4090 because that GPU has a different memory consistency model.

## Key Difference in Generated CUDA Code

### CORRECT version (no T.clear on shared memory):
The pipeline immediately starts prefetching data into shared memory:
```c
// Stage 0 prefetch - directly loads data into shared memory
*(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(A + ...);  // load A
*(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(B + ...);  // load B
// pipeline loop begins cleanly
for (int ko = 0; ko < 30; ++ko) { ... }
```

### BUGGY version (T.clear on B_shared before pipeline):
A zeroing loop runs on shared memory BEFORE the pipeline loads data,
with only a single __syncthreads() separating them — insufficient on H100:
```c
// BUG: zeroing B_shared region before pipeline starts
for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(uint4*)(buf_dyn_shmem + ...) = make_uint4(0, 0, 0, 0);  // zero out!
}
// only one __syncthreads() here - not enough on H100
*(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(A + ...);
__syncthreads();
// pipeline loop - but B_shared may still be partially zeroed on H100
for (int ko = 0; ko < 30; ++ko) { ... }
```

## Requirements

- Python 3.10+
- CUDA 12.3+
- TileLang 0.1.8
- **Any NVIDIA GPU** to generate and inspect the CUDA code
- **NVIDIA H100** required to observe incorrect numerical output at runtime

## Setup
```bash
conda create -n tilelang-bugs python=3.12 -y
conda activate tilelang-bugs
pip install torch==2.4.0
pip install tilelang==0.1.8
```

## How to Run
```bash
python reproduce.py
```

## What to Look For

The script prints the generated CUDA source for both versions side by side.

In the **BUGGY version**, look at the code BEFORE the main `for (int ko = ...)` 
loop. You will see:
1. A zeroing loop using `make_uint4(0, 0, 0, 0)` writing to `buf_dyn_shmem`
2. Only a single `__syncthreads()` after it
3. The pipeline immediately starts loading data into the same memory region

On H100, the async memory engine does not guarantee that the zero-writes are
fully visible to subsequent async loads with just one `__syncthreads()`,
causing the data race.

## Note on Hardware

This bug does NOT require an H100 to demonstrate — the generated CUDA code
difference is visible on any GPU. However, the actual wrong numerical output
(79.3% mismatched elements) only manifests on H100 due to its stricter async
memory pipeline behavior.
