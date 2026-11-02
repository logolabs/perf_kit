"""FP8 forward GEMM on block-FP8 weights (DeepSeek-V3 recipe, forward only): activations quantised per 1 x 128 group
(e4m3, fp32 scales), weights used as the checkpoint stores them (e4m3 codes, one fp32 scale per 128 x 128 block),
fp32 accumulation, bf16 out. LOSSY: the activation quantisation adds ~2-3% relative error per GEMM output, so it is
an A/B-gated option, never a default.

CUDA path: vLLM's CUTLASS block-scaled GEMM (`_custom_ops.cutlass_scaled_mm`, sm90) + its per-token-group quant
kernel. Fallback (CPU / no vLLM): dequantise and multiply in the input dtype (= the exact bf16 path).
"""
from __future__ import annotations

import torch

from .dequant import dequant_torch

_V: dict = {}


def _vllm():
    if "ok" not in _V:
        try:
            from vllm import _custom_ops as ops
            from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
            _V.update(ok=True, mm=ops.cutlass_scaled_mm, quant=per_token_group_quant_fp8)
        except Exception as ex:          # noqa: BLE001
            _V.update(ok=False, err=repr(ex)[:200])
    return _V["ok"]


def available() -> bool:
    return _vllm()


def quant_act(x: torch.Tensor):
    """x [T, K] bf16 -> (codes [T, K] e4m3, scales [T, K/128] fp32 column-major), for fp8_gemm."""
    return _V["quant"](x.contiguous(), 128, column_major_scales=True, use_ue8m0=False)


def fp8_gemm(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor, xq=None) -> torch.Tensor:
    """x [T, K] @ dequant(codes [N, K], scale [N/128, K/128])^T -> [T, N] bf16."""
    if x.is_cuda and _vllm() and x.dtype == torch.bfloat16:
        q, s = xq if xq is not None else quant_act(x)
        return _V["mm"](q, codes.t(), out_dtype=torch.bfloat16, scale_a=s, scale_b=scale.t())
    return x @ dequant_torch(codes, scale, x.dtype).t()
