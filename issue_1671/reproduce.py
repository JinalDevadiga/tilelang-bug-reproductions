"""
Issue #1671 - Python 'and'/'or'/'not' operators used on TVM Expr objects
inside a TileLang kernel cause a ValueError at compile time.

GitHub Issue : https://github.com/tile-ai/tilelang/issues/1671
Reported on  : TileLang v0.1.7.post2
Status       : Open (not reproduced on v0.1.8 — likely silently fixed)

─── What the bug is ────────────────────────────────────────────────────────────

TileLang is built on TVM as its compiler backend. Inside a @T.prim_func kernel,
variables like loop indices and symbolic dimensions are TVM Expr objects, not
plain Python integers. Their values are unknown at Python parse time — they only
get resolved when the kernel is compiled and run on the GPU.

Python's built-in 'and', 'or', and 'not' operators work by calling __bool__()
on their operands to get an immediate True or False. TVM Expr objects deliberately
block this: they raise a ValueError because a symbolic expression has no concrete
boolean value yet.

The original file examples/dsa_sparse_finetune/indexer_topk_reducesum.py
contained conditions written as:

    if row_idx < seq_len and col_idx < num_cols:   # ← Python 'and' on Expr

This crashes with:

    ValueError: Cannot use and / or / not operator to Expr,
                hint: use tvm.tir.all / tvm.tir.any instead

Note: the error message says "tvm.tir" because TVM is the compiler layer that
TileLang is built on. TVM ships inside the TileLang package — it is not a
separate dependency you need to install. When you write `import tilelang`, TVM
comes with it.

─── Why 'and' triggers the error but static conditions don't ───────────────────

Static loop bounds like T.serial(128) produce plain Python integers, so 'and'
works fine on them. The crash only happens when at least one operand is a
symbolic TVM Expr, which occurs when:

  - Using T.symbolic() for dynamic/variable-size dimensions
  - Comparing loop indices against symbolic bounds (e.g. row < seq_len)
  - Any expression involving T.Var objects

─── The fix ────────────────────────────────────────────────────────────────────

Replace Python boolean operators with their TVM-native equivalents:

    BUGGY                          FIXED
    ─────────────────────────────────────────────────
    cond_a and cond_b         →    T.tir.all(cond_a, cond_b)
    cond_a or cond_b          →    T.tir.any(cond_a, cond_b)
    not cond                  →    T.tir.Not(cond)
    min(a, b)  (on Expr)      →    T.tir.min(a, b)
    max(a, b)  (on Expr)      →    T.tir.max(a, b)
"""

import tilelang
import tilelang.language as T


# ── Helper: print a section header ───────────────────────────────────────────
def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


# ═════════════════════════════════════════════════════════════════════════════
# BUGGY KERNEL
# Uses Python 'and' to combine two symbolic conditions.
# On v0.1.7.post2 this raises:
#   ValueError: Cannot use and / or / not operator to Expr,
#               hint: use tvm.tir.all / tvm.tir.any instead
# ═════════════════════════════════════════════════════════════════════════════
section("BUGGY kernel — Python 'and' on symbolic Expr")

BLOCK = 64

def make_buggy_kernel(seq_len: int, num_cols: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((seq_len, num_cols), "float32"),
        B: T.Tensor((seq_len, num_cols), "float32"),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK), threads=BLOCK) as (bx,):
            tx = T.get_thread_binding(0)
            row = bx * BLOCK + tx
            for j in T.serial(num_cols):
                # ← BUG: Python 'and' calls __bool__() on the TVM Expr
                #         (row < seq_len), which raises ValueError because
                #         row is a symbolic expression, not a plain int.
                if row < seq_len and j < num_cols:
                    B[row, j] = A[row, j] * 2.0
    return kernel

try:
    kernel_fn = make_buggy_kernel(128, 64)
    buggy_kernel = tilelang.compile(kernel_fn, out_idx=[1], target="cuda")
    print("No error raised — bug is not present in this version of TileLang.")
    print(f"TileLang version installed: {tilelang.__version__}")
    print()
    print("On v0.1.7.post2 this would raise:")
    print("  ValueError: Cannot use and / or / not operator to Expr,")
    print("              hint: use tvm.tir.all / tvm.tir.any instead")
    print()
    print("Generated CUDA source (buggy kernel compiled successfully here):")
    print(buggy_kernel.get_kernel_source())
except ValueError as e:
    print(f"BUG REPRODUCED — ValueError raised at compile time:")
    print(f"  {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# FIXED KERNEL
# Uses T.tir.all() instead of Python 'and'.
# This is the correct way to combine symbolic conditions in TileLang.
# ═════════════════════════════════════════════════════════════════════════════
section("FIXED kernel — T.tir.all() for symbolic conditions")

def make_fixed_kernel(seq_len: int, num_cols: int):
    @T.prim_func
    def kernel(
        A: T.Tensor((seq_len, num_cols), "float32"),
        B: T.Tensor((seq_len, num_cols), "float32"),
    ):
        with T.Kernel(T.ceildiv(seq_len, BLOCK), threads=BLOCK) as (bx,):
            tx = T.get_thread_binding(0)
            row = bx * BLOCK + tx
            for j in T.serial(num_cols):
                # ← FIX: T.tir.all() combines symbolic conditions safely.
                #         It stays as a symbolic expression; it does NOT
                #         call __bool__() and does NOT raise ValueError.
                if T.tir.all(row < seq_len, j < num_cols):
                    B[row, j] = A[row, j] * 2.0
    return kernel

try:
    kernel_fn = make_fixed_kernel(128, 64)
    fixed_kernel = tilelang.compile(kernel_fn, out_idx=[1], target="cuda")
    print("Fixed kernel compiled successfully.")
    print()
    print("Generated CUDA source:")
    print(fixed_kernel.get_kernel_source())
except Exception as e:
    print(f"Unexpected error in fixed kernel: {type(e).__name__}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
section("Summary")
print("""
Inside @T.prim_func kernels, loop indices and symbolic dimensions are TVM
Expr objects. Python's 'and'/'or'/'not' cannot be used on them.

    BUGGY                            FIXED
    ────────────────────────────────────────────────────────
    cond_a and cond_b           →    T.tir.all(cond_a, cond_b)
    cond_a or  cond_b           →    T.tir.any(cond_a, cond_b)
    not cond                    →    T.tir.Not(cond)
    min(expr_a, expr_b)         →    T.tir.min(expr_a, expr_b)
    max(expr_a, expr_b)         →    T.tir.max(expr_a, expr_b)

Why does the error mention 'tvm.tir'?
TileLang is built on TVM as its compiler backend. TVM ships inside the
TileLang package — you do not install TVM separately. The Expr type,
the ValueError, and the tir.all/tir.any fix are all part of TVM's
internals that TileLang exposes through T.tir.
""")