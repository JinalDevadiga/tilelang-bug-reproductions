# TileLang Bug Reproductions

This repository contains reproductions of bugs found in the TileLang GPU kernel compiler,
including data races, synchronization issues, and compile-time errors.

## Bugs

| Issue | Title | Bug Type | Reproduced On | Status |
|-------|-------|----------|---------------|--------|
| [#1257](issue_1257/) | Missing `__syncthreads()` after `AtomicAdd` | Data race in generated CUDA kernel | v0.1.6 | Fixed in v0.1.8 |
| [#96](issue_96) | Race condition in tutorial matmul | Async pipeline race with `num_stages > 0` | v0.1.5 | Closed |
| [#666](issue_666) | Wrong results clearing shared memory before pipelined loop on H100 | Sync bug between `T.clear` and async pipeline stages | v0.1.5 | Closed |
| [#1671](issue_1671) | Python `and`/`or`/`not` on TVM Expr raises `ValueError` | Compile-time error from incorrect use of Python boolean operators on symbolic expressions | v0.1.7.post2 | Open (not reproduced on v0.1.8) |

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
│   ├── README.md
│   └── reproduce.py
├── issue_96_pipeline_race/
│   ├── README.md
│   └── reproduce.py
├── issue_666_clear_before_pipeline/
│   ├── README.md
│   └── reproduce.py
└── issue_1671_and_or_in_expr/
    ├── README.md
    └── reproduce.py
```

## Notes

- Issues #96 and #666 require an NVIDIA GPU with sufficient memory. Issue #666 originally only
  reproduced on an H100 — the `reproduce.py` for that issue generates the buggy CUDA source code
  without needing to execute it on hardware, following mentor guidance.
- Issue #1671 is a compile-time error rather than a data race. It is included for completeness
  as it was part of the collected issue set.
- All scripts print the generated CUDA kernel source so the bug is visible without needing
  specific hardware.