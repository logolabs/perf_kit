"""Ideogram 4's weight-only FP8 layout (e4m3 codes, ONE float32 scale per output row) -> bf16, and the optional
FP8 forward GEMM.

`dequant_rows_into(dst, codes, scale)` writes (codes.float() * scale[:, None]).to(bf16): bit for bit what a
dequantise-at-load trainer stores, so keeping the checkpoint as codes and dequantising right before each GEMM
computes with the SAME bf16 weights in half the memory.

`fp8_linear(x, codes, scale)` (lossy, measured and NOT recommended): the weight codes are used as is (they are the
checkpoint) and the activations are quantised per token (row-wise e4m3, amax / 448), then torch._scaled_mm with
row-wise scales. On GH200 it gave no speed-up and 10x the reference's loss rounding error (internal, 2026-10-31).
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0

try:                                    # triton is optional (CPU-only installs, Windows test boxes)
    import triton
    import triton.language as tl

    @triton.jit
    def _dq_rows_kernel(c_ptr, s_ptr, o_ptr, N, K, stride_o, BN: tl.constexpr, BK: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = pid_k * BK + tl.arange(0, BK)
        m = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(c_ptr + rn[:, None] * K + rk[None, :], mask=m, other=0.0).to(tl.float32)
        s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
        tl.store(o_ptr + rn[:, None] * stride_o + rk[None, :], (w * s[:, None]).to(tl.bfloat16), mask=m)

    HAVE_TRITON = True
except Exception:                       # noqa: BLE001
    HAVE_TRITON = False


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A 2-D weight -> (e4m3 codes [out, in], float32 scale [out]) with scale = row amax / 448: the layout of
    Ideogram 4's FP8 checkpoint (used by the tests to build FP8 layers; real checkpoints arrive quantised)."""
    wf = w.detach().to(torch.float32)
    amax = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(1).to(torch.float32)


def dequant_rows_torch(codes: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    return (codes.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).to(dtype)


def dequant_rows_into(dst: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = codes.shape
    if tuple(dst.shape) != (n, k):
        raise ValueError(f"dequant_rows_into: dst {tuple(dst.shape)} != codes {(n, k)}")
    if (codes.is_cuda and HAVE_TRITON and dst.dtype == torch.bfloat16 and codes.is_contiguous()
            and dst.stride(1) == 1 and scale.is_contiguous() and scale.dtype == torch.float32):
        BN, BK = 32, 256
        grid = (triton.cdiv(n, BN), triton.cdiv(k, BK))
        _dq_rows_kernel[grid](codes, scale, dst, n, k, dst.stride(0), BN=BN, BK=BK)
        return dst
    dst.copy_(dequant_rows_torch(codes, scale, dst.dtype))
    return dst


def quant_act_rows(x: torch.Tensor):
    """x [M, K] -> (e4m3 codes [M, K], float32 scale [M, 1]) with scale = amax / 448 per row."""
    amax = x.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12)
    s = amax / FP8_MAX
    q = (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


def fp8_linear(x: torch.Tensor, codes: torch.Tensor, scale_row: torch.Tensor, out_dtype=torch.bfloat16):
    """x [M, K] bf16, codes [N, K] e4m3 (row-major), scale_row [1, N] float32 -> x W^T [M, N] (lossy)."""
    xq, sx = quant_act_rows(x)
    return torch._scaled_mm(xq, codes.t(), scale_a=sx, scale_b=scale_row, out_dtype=out_dtype)
