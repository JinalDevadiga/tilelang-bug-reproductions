# Issue #1671 — Python `and`/`or`/`not` on TVM Expr Inside TileLang Kernel

## Source
- **GitHub Issue:** https://github.com/tile-ai/tilelang/issues/1671
- **Repo:** TileLang
- **Reported on:** v0.1.7.post2
- **Status:** Open (not reproduced on v0.1.8 — likely silently fixed)

---

## What is the Bug?

Running `examples/dsa_sparse_finetune/indexer_topk_reducesum.py` on
TileLang v0.1.7.post2 raises this error at compile time:
```
ValueError: Cannot use and / or / not operator to Expr,
            hint: use tvm.tir.all / tvm.tir.any instead
```

The cause is Python's `and`/`or`/`not` operators being used on TVM
symbolic expressions inside a `@T.prim_func` kernel. Python calls
`__bool__()` on both sides of `and`/`or` to get an immediate True/False,
but TVM Expr objects block this because their values are not known until
the kernel is compiled and run on the GPU.

---

## Why Does the Error Mention `tvm.tir`?

TileLang is built on **TVM** as its compiler backend. TVM ships
**inside** the TileLang package — it is not a separate dependency.
When you install `tilelang`, TVM comes with it. The `Expr` type, the
`ValueError`, and the `tir.all`/`tir.any` fix all come from TVM's
internals, which TileLang exposes as `T.tir`.

---

## Root Cause

Inside a `@T.prim_func` kernel, variables like loop indices and symbolic
dimensions are **TVM Expr objects**, not plain Python integers. Their
values are only resolved at kernel compile/run time on the GPU.

Python's `and`/`or`/`not` keywords work by calling `__bool__()` on their
operands to evaluate them as True/False immediately. TVM's `Expr.__bool__`
is intentionally blocked and raises `ValueError`.

This only happens with **symbolic expressions** — for example:
```python
row = bx * BLOCK + tx        # TVM Expr (symbolic)
seq_len = 128                 # plain Python int

if row < seq_len and j < 64:  # ← crashes: 'row < seq_len' is an Expr
```

Static integer conditions (e.g. `if 3 < 10 and 2 < 5`) are fine because
Python evaluates those immediately without producing a TVM Expr.

---

## The Fix

Replace Python boolean operators with their TVM-native equivalents:

| Buggy | Fixed |
|-------|-------|
| `cond_a and cond_b` | `T.tir.all(cond_a, cond_b)` |
| `cond_a or cond_b` | `T.tir.any(cond_a, cond_b)` |
| `not cond` | `T.tir.Not(cond)` |
| `min(expr_a, expr_b)` | `T.tir.min(expr_a, expr_b)` |
| `max(expr_a, expr_b)` | `T.tir.max(expr_a, expr_b)` |

---

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- `tilelang` (any version)
- `torch`
```bash
pip install tilelang torch
```

---

## How to Run
```bash
python reproduce.py
```

---

## Expected Output

On **v0.1.7.post2** (the version where the bug was reported), the buggy
kernel section would raise:
```
ValueError: Cannot use and / or / not operator to Expr,
            hint: use tvm.tir.all / tvm.tir.any instead
```

On **v0.1.8+** the buggy kernel compiles without error, and the script
prints the generated CUDA source for both kernels. The fixed kernel
compiles and runs correctly on both versions.

---

## Note on Version Behaviour

This bug could not be reproduced on v0.1.8. The issue remains open on
GitHub with no linked fix or closing commit, so it is unclear exactly
when or how it was resolved. The `reproduce.py` script handles both
cases: it reports whether the error is raised or not, and always
demonstrates the correct fix using `T.tir.all()`.