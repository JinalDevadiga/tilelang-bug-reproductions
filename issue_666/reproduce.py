import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout as make_swizzle_layout


def matmul_correct(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """Correct version - T.clear only on local fragment, NOT on shared memory before pipeline"""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)  # OK - clearing local fragment before pipeline

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


def matmul_buggy(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """Buggy version - T.clear on shared memory BEFORE pipelined loop"""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(B_shared)  # BUG - clearing shared memory before pipeline
                                # causes sync issue with async pipeline stages on H100

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32

print("=== Issue #666: Incorrect Results When Clearing Shared Memory")
print("=== Before Pipelined Loops on NVIDIA H100")
print()
print("NOTE: The incorrect OUTPUT only manifests on H100 GPUs.")
print("      This script demonstrates the bug by showing the difference")
print("      in generated CUDA code between the correct and buggy versions.")
print()

print("--- Compiling CORRECT version (T.clear only on local fragment) ---")
func_correct = matmul_correct(M, N, K, block_M, block_N, block_K)
kernel_correct = tilelang.compile(func_correct, out_idx=[2], target="cuda")
print("Generated CUDA source (CORRECT):")
print(kernel_correct.get_kernel_source())
print()

print("--- Compiling BUGGY version (T.clear on shared memory before pipeline) ---")
func_buggy = matmul_buggy(M, N, K, block_M, block_N, block_K)
kernel_buggy = tilelang.compile(func_buggy, out_idx=[2], target="cuda")
print("Generated CUDA source (BUGGY):")
print(kernel_buggy.get_kernel_source())
print()

print("--- What to look for ---")
print("In the BUGGY version, look for a memset/clear of B_shared BEFORE")
print("the pipelined async copy loop begins. On H100, the async pipeline")
print("stages race against this clear operation causing 79%% wrong output.")
print("On other GPUs (RTX 4090 etc.) this may not cause visible errors.")
