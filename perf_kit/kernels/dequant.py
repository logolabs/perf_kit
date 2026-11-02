"""Block-FP8 (E4M3, one fp32 scale per 128 x 128 block, the DeepSeek-V3 / Qwen-FP8 checkpoint layout) -> bf16.

`dequant_into(dst, w, scale_inv)` writes w * scale (computed in fp32, rounded once to bf16) into `dst`, which may be
a row slice of a larger buffer. That is bit-for-bit what loading the checkpoint as `(w.float() * s).to(bf16)` gives,
so a model that keeps its frozen weights as FP8 codes and dequantises them just before each GEMM computes with the
SAME bf16 weights as one that stored them dequantised (half the memory, no approximation).

CUDA: one Triton program per 128 x 128 block. Elsewhere (CPU tests): the torch expression.
"""
from __future__ import annotations

import torch

BLOCK = 128

try:                                    # triton is optional (CPU-only installs, Windows test boxes)
    import triton
    import triton.language as tl

    @triton.jit
    def _dequant_kernel(w_ptr, s_ptr, o_ptr, N, K, KB, stride_o, BLOCK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BLOCK + tl.arange(0, BLOCK)
        rk = pid_k * BLOCK + tl.arange(0, BLOCK)
        m = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=m, other=0.0).to(tl.float32)
        sc = tl.load(s_ptr + pid_n * KB + pid_k)
        tl.store(o_ptr + rn[:, None] * stride_o + rk[None, :], (w * sc).to(tl.bfloat16), mask=m)

    HAVE_TRITON = True
except Exception:                       # noqa: BLE001
    HAVE_TRITON = False


def expand_scale(scale_inv: torch.Tensor, n: int, k: int, block: int = BLOCK) -> torch.Tensor:
    return scale_inv.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]


def dequant_torch(w: torch.Tensor, scale_inv: torch.Tensor, dtype=torch.bfloat16, block: int = BLOCK) -> torch.Tensor:
    return (w.float() * expand_scale(scale_inv, w.shape[0], w.shape[1], block)).to(dtype)


def dequant_into(dst: torch.Tensor, w: torch.Tensor, scale_inv: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """dst[:] = dequant(w); dst is [N, K] with unit column stride (a row slice of a bigger buffer is fine)."""
    n, k = w.shape
    if dst.shape != (n, k):
        raise ValueError(f"dequant_into: dst {tuple(dst.shape)} != w {(n, k)}")
    if (w.is_cuda and HAVE_TRITON and dst.dtype == torch.bfloat16 and block == BLOCK and w.is_contiguous()
            and dst.stride(1) == 1 and scale_inv.is_contiguous()):
        sc = scale_inv if scale_inv.dtype == torch.float32 else scale_inv.float()
        grid = (triton.cdiv(n, block), triton.cdiv(k, block))
        _dequant_kernel[grid](w, sc, dst, n, k, sc.shape[1], dst.stride(0), BLOCK=block)
        return dst
    dst.copy_(dequant_torch(w, scale_inv, dst.dtype, block))
    return dst
