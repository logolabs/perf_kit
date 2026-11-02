"""The non-GEMM maths of an Ideogram 4 block on PACKED tokens, forward and hand-written backward.

Every function reproduces the numerics of the reference (the upstream modules run under torch.autocast(bfloat16),
whose policy runs rms_norm / layer_norm in float32 and leaves silu / mul in the input dtype):
  * RMSNorm / LayerNorm in float32, weight in float32, the result rounded to the compute dtype only where the
    reference feeds a GEMM;
  * the residual stream in float32 (the reference's `x + gate * norm(...)` promotes to float32 from block 0 on);
  * SwiGLU in the compute dtype with torch's own rounding (silu rounded, then times the up projection);
  * rotary: float32 maths with cos / sin rounded to the compute dtype (upstream `cos.to(h.dtype)`).

Per-sample modulation tensors are (B, D); `seg` maps each packed token to its sample. With `set_compile(True)`
each function is wrapped in torch.compile (inductor fuses the gather, the norm and the cast into one kernel);
the maths is unchanged, inductor's fusion only re-orders float32 rounding. Attention and the varlen layout never
enter dynamo (they are called between these functions), which is what makes compiling this model work at all.

With `set_triton_swiglu(True)` SwiGLU runs on perf_kit's Triton kernels (perf_kit/kernels/ops.py: torch's bf16
rounding, forward bitwise and backward within one bf16 ulp of the eager path on GH200). Inductor's fused kernel for
the same expression ran ~5x below memory bandwidth on GH200 (internal measurement, 2026-10-31).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_COMPILED: dict = {}
_STATE = {"compile": False, "triton_swiglu": False}


def set_compile(on: bool) -> None:
    _STATE["compile"] = bool(on)


def set_triton_swiglu(on: bool) -> None:
    _STATE["triton_swiglu"] = bool(on)


def _maybe(fn):
    """Dispatch to a compiled copy when compile is on (one compiled copy per function, dynamic shapes)."""
    name = fn.__name__

    def wrapper(*a, **kw):
        if _STATE["compile"]:
            c = _COMPILED.get(name)
            if c is None:
                c = _COMPILED[name] = torch.compile(fn, dynamic=True)
            return c(*a, **kw)
        return fn(*a, **kw)

    wrapper.__name__ = name
    wrapper.eager = fn
    return wrapper


def _rstd(xf: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _rms_bwd(xf: torch.Tensor, g: torch.Tensor, eps: float) -> torch.Tensor:
    """d/dx of x * rstd(x) given g = dy * w (all float32 / float64)."""
    r = _rstd(xf, eps)
    return r * (g - xf * (r * r) * (g * xf).mean(-1, keepdim=True))


# ------------------------------------------------------------------ norm + modulation (GEMM input)
@_maybe
def norm_mod(x, w, mod, seg, eps: float, out_dtype: torch.dtype):
    """(rmsnorm(x) * w * mod[seg]).to(out_dtype); x [T, D] float32, w [D], mod [B, D]."""
    xf = x.float() if x.dtype != torch.float64 else x
    y = xf * _rstd(xf, eps) * w.to(xf.dtype)
    y = y * mod[seg].to(xf.dtype)
    return y.to(out_dtype)


@_maybe
def norm_mod_bwd(x, w, mod, seg, dy, eps: float):
    """dx (float32) of norm_mod for dy [T, D] (the GEMM input's gradient)."""
    xf = x.float() if x.dtype != torch.float64 else x
    g = dy.to(xf.dtype) * mod[seg].to(xf.dtype) * w.to(xf.dtype)
    return _rms_bwd(xf, g, eps)


# ------------------------------------------------------------------ gated residual: x + gate[seg] * rmsnorm(a) * w
@_maybe
def gate_res(x, a, w, gate, seg, eps: float):
    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    af = a.to(xf.dtype)
    return xf + gate[seg].to(xf.dtype) * (af * _rstd(af, eps) * w.to(xf.dtype))


@_maybe
def gate_res_bwd(a, w, gate, seg, dx, eps: float):
    """d a (in a's dtype) of gate_res for the residual gradient dx (float32); dx itself passes straight through."""
    af = a.to(dx.dtype)
    g = dx * gate[seg].to(dx.dtype) * w.to(dx.dtype)
    return _rms_bwd(af, g, eps).to(a.dtype)


# ------------------------------------------------------------------ q / k RMSNorm per head + rotary
def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _rot_t(g):
    """Transpose of _rot: <g, rot(x)> = <rot_t(g), x>."""
    h = g.shape[-1] // 2
    return torch.cat((g[..., h:], -g[..., :h]), dim=-1)


@_maybe
def qk_norm_rope(qkv, wq, wk, cos, sin, heads: int, eps: float):
    """qkv [T, 3D] -> q, k (rotated, compute dtype) [T, H, hd] and v [T, H, hd] (a view)."""
    T = qkv.shape[0]
    D = qkv.shape[1] // 3
    hd = D // heads
    q, k, v = qkv.view(T, 3, heads, hd).unbind(1)
    ft = torch.float64 if qkv.dtype == torch.float64 else torch.float32
    c, s = cos.to(ft)[:, None, :], sin.to(ft)[:, None, :]
    outs = []
    for t, w in ((q, wq), (k, wk)):
        tf = t.to(ft)
        n = tf * _rstd(tf, eps) * w.to(ft)
        outs.append((n * c + _rot(n) * s).to(qkv.dtype))
    return outs[0], outs[1], v


@_maybe
def qk_norm_rope_bwd(qkv, wq, wk, cos, sin, dq, dk, dv, heads: int, eps: float):
    """dqkv [T, 3D] (compute dtype) from the attention input gradients dq, dk, dv [T, H, hd]."""
    T = qkv.shape[0]
    D = qkv.shape[1] // 3
    hd = D // heads
    q, k, _ = qkv.view(T, 3, heads, hd).unbind(1)
    ft = torch.float64 if qkv.dtype == torch.float64 else torch.float32
    c, s = cos.to(ft)[:, None, :], sin.to(ft)[:, None, :]
    grads = []
    for t, w, d in ((q, wq, dq), (k, wk, dk)):
        g = d.to(ft)
        dn = g * c + _rot_t(g * s)                      # through the rotary
        grads.append(_rms_bwd(t.to(ft), dn * w.to(ft), eps).to(qkv.dtype))
    return torch.stack((grads[0], grads[1], dv.to(qkv.dtype)), dim=1).view(T, 3 * D)


# ------------------------------------------------------------------ SwiGLU on the fused [w1 | w3] output
@_maybe
def _swiglu(g13, inter: int):
    return F.silu(g13[:, :inter]) * g13[:, inter:]


@_maybe
def _swiglu_bwd(g13, dh, inter: int):
    a, b = g13[:, :inter], g13[:, inter:]
    s = F.silu(a)
    db = dh * s
    da = torch.ops.aten.silu_backward(dh * b, a)
    return torch.cat((da, db), dim=1)


def _triton_ok(x: torch.Tensor) -> bool:
    if not (_STATE["triton_swiglu"] and x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(1) == 1):
        return False
    from ...kernels import ops as K
    return K._HAVE_TRITON


_BLOCK = 1024


def swiglu(g13, inter: int):
    if _triton_ok(g13):
        from ...kernels import ops as K
        T = g13.shape[0]
        out = torch.empty(T, inter, dtype=g13.dtype, device=g13.device)
        K._swiglu_fwd[(T, (inter + _BLOCK - 1) // _BLOCK)](g13, out, inter, g13.stride(0), out.stride(0), BLOCK=_BLOCK)
        return out
    return _swiglu(g13, inter)


def swiglu_bwd(g13, dh, inter: int):
    if _triton_ok(g13):
        from ...kernels import ops as K
        dh = dh.contiguous()
        T = g13.shape[0]
        dg = torch.empty(T, 2 * inter, dtype=g13.dtype, device=g13.device)
        K._swiglu_bwd[(T, (inter + _BLOCK - 1) // _BLOCK)](g13, dh, dg, inter, g13.stride(0), dh.stride(0),
                                                           dg.stride(0), BLOCK=_BLOCK)
        return dg
    return _swiglu_bwd(g13, dh, inter)


# ------------------------------------------------------------------ final layer: LayerNorm (no affine) * scale
@_maybe
def final_norm(x, scale, seg, eps: float, out_dtype: torch.dtype):
    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    r = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)
    return (xc * r * scale[seg].to(xf.dtype)).to(out_dtype)


@_maybe
def final_norm_bwd(x, scale, seg, dy, eps: float):
    xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
    mu = xf.mean(-1, keepdim=True)
    xc = xf - mu
    r = torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + eps)
    xh = xc * r
    g = dy.to(xf.dtype) * scale[seg].to(xf.dtype)
    return r * (g - g.mean(-1, keepdim=True) - xh * (g * xh).mean(-1, keepdim=True))


# ------------------------------------------------------------------ plain float32 RMSNorm with weight (llm_cond_norm)
def rmsnorm_w(x, w, eps: float, out_dtype: torch.dtype):
    xf = x.float() if x.dtype != torch.float64 else x
    return (xf * _rstd(xf, eps) * w.to(xf.dtype)).to(out_dtype)
