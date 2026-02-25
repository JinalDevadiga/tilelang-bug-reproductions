# TileLang Bug Reproductions

This repository contains reproductions of data race bugs found in the TileLang GPU kernel compiler.

## Bugs

| Issue | Title | Bug Type | Status |
|-------|-------|----------|--------|
| [#1257](issue_1257/) | Missing `__syncthreads()` after `AtomicAdd` | Data race in generated CUDA kernel | Fixed in v0.1.8 |
<!-- | [#96](issue_96_pipeline_race/) | Race condition in tutorial matmul | Async pipeline race | Closed |
| [#666](issue_666_clear_before_pipeline/) | Wrong results clearing shared memory on H100 | Sync bug with pipelined loops | Closed | -->

## Setup
```bash
conda create -n tilelang-bugs python=3.10 -y
conda activate tilelang-bugs
pip install torch==2.4.0
pip install tilelang
```

## Repository Structure
```
tilelang-bug-reproductions/
├── README.md
├── issue_1257/
    ├── README.md
    └── reproduce.py
```
