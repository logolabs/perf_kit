"""The non-GEMM ops of a hybrid (linear-attention + softmax-attention) decoder on PACKED sequences, each with a fast
CUDA path and a pure-torch reference that reproduces Hugging Face's math (the CPU exactness tests compare the
references, in float64, against the transformers model itself).

  rmsnorm         Liger's Triton RMSNorm (gemma casting, offset 1: Qwen3.5 / Qwen3-Next / Gemma)  | HF math
  silu_mul        Triton SwiGLU reading the fused [gate | up] projection in place, writing d[gate | up] contiguous
                  for the projection's backward (no slicing copies)                                  | HF math
  conv_silu       fla's Triton causal_conv1d with cu_seqlens (rows never see each other)             | per-row conv
  gdn             fla chunk_gated_delta_rule, native GVA (no q/k repeat), varlen                      | per-row HF torch
  attention       flash-attn 2 varlen, causal, GQA                                                    | per-row SDPA
Backend switches (experiment copies; logged through available()): PK_GDN_BACKEND = fla (default) | flashqla (Qwen's
TileLang GDN kernels; q/k L2-normalised and g/beta computed in torch first), PK_ATTN_BACKEND = fa2 (default) | fa3
(flash-attn 3 varlen, Hopper).
  gated_rmsnorm   fla's fused gated RMSNorm (swish gate)                                              | HF math

`impl` is "fast" or "ref"; "auto" = fast on CUDA when the library imports, else ref.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

_LIBS: dict = {}
GDN_BACKEND = os.environ.get("PK_GDN_BACKEND", "fla")
ATTN_BACKEND = os.environ.get("PK_ATTN_BACKEND", "fa2")
if GDN_BACKEND not in ("fla", "flashqla") or ATTN_BACKEND not in ("fa2", "fa3"):
    raise ValueError(f"PK_GDN_BACKEND={GDN_BACKEND!r} / PK_ATTN_BACKEND={ATTN_BACKEND!r}")


def _lib(name):
    if name not in _LIBS:
        try:
            if name == "liger_rms":
                from liger_kernel.ops.rms_norm import LigerRMSNormFunction as m
            elif name == "liger_add_rms":
                from liger_kernel.ops.fused_add_rms_norm import LigerFusedAddRMSNormFunction as m
            elif name == "fla_conv":
                from fla.modules.conv import causal_conv1d as m
            elif name == "fla_gdn":
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule as m
            elif name == "fla_gnorm":
                from fla.modules.fused_norm_gate import rms_norm_gated as m
            elif name == "flash":
                from flash_attn import flash_attn_varlen_func as m
            elif name == "fa3":
                from flash_attn_interface import flash_attn_varlen_func as m
            elif name == "flashqla":
                from flash_qla import chunk_gated_delta_rule as m
            elif name == "triton":
                import triton as m
            else:
                raise KeyError(name)
            _LIBS[name] = m
        except Exception:               # noqa: BLE001
            _LIBS[name] = None
    return _LIBS[name]


def use_fast(impl: str, x: torch.Tensor, lib: str) -> bool:
    if impl == "ref":
        return False
    ok = x.is_cuda and _lib(lib) is not None
    if impl == "fast" and not ok:
        raise RuntimeError(f"fast path needs CUDA and {lib}")
    return ok


def available() -> dict:
    out = {k: _lib(k) is not None for k in ("liger_rms", "liger_add_rms", "fla_conv", "fla_gdn", "fla_gnorm", "flash",
                                            "triton", "fa3", "flashqla")}
    return {**out, "gdn_backend": GDN_BACKEND, "attn_backend": ATTN_BACKEND}


# ------------------------------------------------------------------ RMSNorm, (1 + w) offset, fp32 compute

def rmsnorm_ref(x, w, eps):
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + w.float())
    return out.type_as(x)


def rmsnorm(x, w, eps, impl="auto"):
    if use_fast(impl, x, "liger_rms"):
        shp = x.shape
        return _lib("liger_rms").apply(x.reshape(-1, shp[-1]).contiguous(), w, eps, 1.0, "gemma", False).view(shp)
    return rmsnorm_ref(x, w, eps)


def rmsnorm_llama_ref(x, w, eps):
    """Llama / Qwen2 / Qwen3 / Mistral: fp32 normalisation, cast back, THEN times the weight."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rmsnorm_llama(x, w, eps, impl="auto"):
    if use_fast(impl, x, "liger_rms"):
        shp = x.shape
        return _lib("liger_rms").apply(x.reshape(-1, shp[-1]).contiguous(), w, eps, 0.0, "llama", False).view(shp)
    return rmsnorm_llama_ref(x, w, eps)


def add_rmsnorm(delta, h, w, eps, style="offset", impl="auto"):
    """(norm(h + delta), h + delta): the residual add fused into the next RMSNorm (one kernel each way).
    style "offset": Qwen3.5 / Gemma ((1 + w), all fp32); "llama": w * norm(x) cast back first."""
    if use_fast(impl, h, "liger_add_rms"):
        shp = h.shape
        y, s = _lib("liger_add_rms").apply(delta.reshape(-1, shp[-1]).contiguous(), h.reshape(-1, shp[-1]).contiguous(),
                                         w, eps, 1.0 if style == "offset" else 0.0,
                                         "gemma" if style == "offset" else "llama", False)
        return y.view(shp), s.view(shp)
    s = h + delta
    return (rmsnorm_ref(s, w, eps) if style == "offset" else rmsnorm_llama_ref(s, w, eps)), s


# ------------------------------------------------------------------ SwiGLU on the fused [gate | up] output

def silu_mul_ref(gu, inter):
    return F.silu(gu[..., :inter]) * gu[..., inter:]


try:                                    # module-level kernels: triton reads their source from this file
    import triton
    import triton.language as tl

    @triton.jit
    def _swiglu_fwd(gu_ptr, o_ptr, I, stride_gu, stride_o, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < I
        g = tl.load(gu_ptr + row * stride_gu + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * stride_gu + I + cols, mask=m, other=0.0).to(tl.float32)
        s = (g * tl.sigmoid(g)).to(tl.bfloat16).to(tl.float32)        # torch: silu rounded to bf16, then * up
        tl.store(o_ptr + row * stride_o + cols, (s * u).to(tl.bfloat16), mask=m)

    @triton.jit
    def _swiglu_bwd(gu_ptr, do_ptr, dgu_ptr, I, stride_gu, stride_do, stride_dgu, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = cols < I
        g = tl.load(gu_ptr + row * stride_gu + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(gu_ptr + row * stride_gu + I + cols, mask=m, other=0.0).to(tl.float32)
        do = tl.load(do_ptr + row * stride_do + cols, mask=m, other=0.0).to(tl.float32)
        sg = tl.sigmoid(g)
        s = (g * sg).to(tl.bfloat16).to(tl.float32)
        du = (do * s).to(tl.bfloat16)                                  # d(up) = do * silu(g)
        ds = (do * u).to(tl.bfloat16).to(tl.float32)                   # d(silu out) = do * up, rounded as torch
        dg = (ds * sg * (1.0 + g * (1.0 - sg))).to(tl.bfloat16)        # silu backward in fp32
        tl.store(dgu_ptr + row * stride_dgu + cols, dg, mask=m)
        tl.store(dgu_ptr + row * stride_dgu + I + cols, du, mask=m)

    _HAVE_TRITON = True
except Exception:                       # noqa: BLE001
    _HAVE_TRITON = False


class _SiluMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gu, inter):
        T = gu.shape[0]
        out = torch.empty(T, inter, dtype=gu.dtype, device=gu.device)
        B = 1024
        _swiglu_fwd[(T, (inter + B - 1) // B)](gu, out, inter, gu.stride(0), out.stride(0), BLOCK=B)
        ctx.save_for_backward(gu)
        ctx.inter = inter
        return out

    @staticmethod
    def backward(ctx, do):
        (gu,) = ctx.saved_tensors
        do = do.contiguous()
        T, inter = gu.shape[0], ctx.inter
        dgu = torch.empty(T, 2 * inter, dtype=gu.dtype, device=gu.device)
        B = 1024
        _swiglu_bwd[(T, (inter + B - 1) // B)](gu, do, dgu, inter, gu.stride(0), do.stride(0), dgu.stride(0), BLOCK=B)
        return dgu, None


def silu_mul(gu, inter, impl="auto"):
    """gu: [T, 2 * inter] (row stride free, unit column stride) -> [T, inter]."""
    if (_HAVE_TRITON and gu.dtype == torch.bfloat16 and gu.dim() == 2 and gu.stride(1) == 1
            and use_fast(impl, gu, "triton")):
        return _SiluMul.apply(gu, inter)
    return silu_mul_ref(gu, inter)


# ------------------------------------------------------------------ causal depthwise conv + SiLU, varlen

def conv_silu_ref(x, w, cu_cpu):
    """x [T, D], w [D, K]: HF's causal_conv1d_fn per row (zero left padding inside each row)."""
    outs = []
    K = w.shape[1]
    for s, e in zip(cu_cpu[:-1].tolist(), cu_cpu[1:].tolist()):
        xr = x[s:e].t()[None]                                          # [1, D, L]
        o = F.conv1d(xr.to(w.dtype), w.unsqueeze(1), None, padding=K - 1, groups=w.shape[0])[..., : e - s]
        outs.append(F.silu(o)[0].t().to(x.dtype))
    return torch.cat(outs, 0)


def conv_silu(x, w, packed, impl="auto"):
    if use_fast(impl, x, "fla_conv"):
        y, _ = _lib("fla_conv")(x.contiguous()[None], w, activation="silu", cu_seqlens=packed.cu64,
                                cu_seqlens_cpu=packed.cu_cpu)
        return y[0]
    return conv_silu_ref(x, w, packed.cu_cpu)


# ------------------------------------------------------------------ gated delta rule (linear attention), varlen

def gdn_ref(q, k, v, g, beta, packed):
    """q, k [T, H, K]; v [T, HV, V]; g, beta [T, HV]. HF's torch chunked kernel per row (q/k repeated to HV)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as ref
    rep = v.shape[1] // q.shape[1]
    outs = []
    for s, e in zip(packed.cu_cpu[:-1].tolist(), packed.cu_cpu[1:].tolist()):
        qq, kk = q[s:e][None], k[s:e][None]
        if rep > 1:
            qq, kk = qq.repeat_interleave(rep, dim=2), kk.repeat_interleave(rep, dim=2)
        o, _ = ref(qq, kk, v[s:e][None], g=g[s:e][None], beta=beta[s:e][None], initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=True)
        outs.append(o[0])
    return torch.cat(outs, 0)


def l2norm(x, eps=1e-6):
    """fla's l2norm (use_qk_l2norm_in_kernel): x / sqrt(sum x^2 + eps), fp32 compute."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


def gdn(q, k, v, g, beta, packed, impl="auto"):
    if GDN_BACKEND == "flashqla" and use_fast(impl, q, "flashqla"):
        o, _ = _lib("flashqla")(l2norm(q)[None], l2norm(k)[None], v[None], g[None].float(), beta[None],
                                scale=q.shape[-1] ** -0.5, initial_state=None, output_final_state=False,
                                cu_seqlens=packed.cu64)
        return o[0]
    if use_fast(impl, q, "fla_gdn"):
        o, _ = _lib("fla_gdn")(q[None], k[None], v[None], g=g[None], beta=beta[None], use_qk_l2norm_in_kernel=True,
                               cu_seqlens=packed.cu64, cu_seqlens_cpu=packed.cu_cpu)
        return o[0]
    return gdn_ref(q, k, v, g, beta, packed)


def gdn_gated(q, k, v, a, b, A_log, dt_bias, packed, impl="auto", fused_gate=True):
    """Qwen3.5 / Qwen3-Next Gated DeltaNet from the RAW projections: g = -exp(A_log) * softplus(a + dt_bias),
    beta = sigmoid(b). Fast path: fla computes both inside its kernels (use_gate_in_kernel /
    use_beta_sigmoid_in_kernel; ~10 fewer launches per layer per direction)."""
    if fused_gate and GDN_BACKEND == "fla" and use_fast(impl, q, "fla_gdn"):
        o, _ = _lib("fla_gdn")(q[None], k[None], v[None], g=a[None], beta=b[None], use_qk_l2norm_in_kernel=True,
                               use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                               use_beta_sigmoid_in_kernel=True, cu_seqlens=packed.cu64, cu_seqlens_cpu=packed.cu_cpu)
        return o[0]
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    return gdn(q, k, v, g, b.sigmoid(), packed, impl)


# ------------------------------------------------------------------ softmax attention, varlen, causal, GQA

def attention_ref(q, k, v, packed, scale):
    """q [T, Hq, D], k/v [T, Hkv, D] -> [T, Hq, D]; per row, HF's eager math (GQA by repeating k/v). (Not SDPA:
    torch 2.5's CPU SDPA returns NaN for float64 causal rows longer than ~32.)"""
    rep = q.shape[1] // k.shape[1]
    outs = []
    for s, e in zip(packed.cu_cpu[:-1].tolist(), packed.cu_cpu[1:].tolist()):
        qq = q[s:e].transpose(0, 1)
        kk = k[s:e].transpose(0, 1).repeat_interleave(rep, dim=0)
        vv = v[s:e].transpose(0, 1).repeat_interleave(rep, dim=0)
        w = (qq @ kk.transpose(1, 2)) * scale
        L = e - s
        w = w.masked_fill(torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        outs.append((w @ vv).transpose(0, 1))
    return torch.cat(outs, 0)


def attention(q, k, v, packed, scale, impl="auto"):
    if ATTN_BACKEND == "fa3" and use_fast(impl, q, "fa3"):
        o = _lib("fa3")(q, k, v, packed.cu, packed.cu, packed.max_len, packed.max_len, softmax_scale=scale, causal=True)
        return o[0] if isinstance(o, tuple) else o
    if use_fast(impl, q, "flash"):
        return _lib("flash")(q, k, v, packed.cu, packed.cu, packed.max_len, packed.max_len, causal=True,
                             softmax_scale=scale)
    return attention_ref(q, k, v, packed, scale)


# ------------------------------------------------------------------ gated RMSNorm (GDN output norm)

def gated_rmsnorm_ref(x, z, w, eps):
    """HF Qwen3_5RMSNormGated: w * norm(x) (cast back first), times silu(z) in fp32, cast to x's dtype."""
    dt = x.dtype
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    h = w * h.to(dt)
    h = h * F.silu(z.float())
    return h.to(dt)


def gated_rmsnorm(x, z, w, eps, impl="auto"):
    if use_fast(impl, x, "fla_gnorm"):
        return _lib("fla_gnorm")(x.contiguous(), z.contiguous(), w, None, "swish", eps=eps)
    return gated_rmsnorm_ref(x, z, w, eps)
