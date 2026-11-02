"""Kernel microbenchmarks at a hybrid LLM's real shapes (default: Qwen3.8-27B, hidden 5120, 64 layers = 48 Gated
DeltaNet + 16 gated attention), one GPU. Each case runs in its own try block and lands in one JSON; a failing case
records its error and the rest still run. Cases (bf16; T = packed tokens per micro-batch):

  launch      per-launch CPU cost of a tiny kernel, eager vs CUDA-graph replay (is the step launch-bound?)
  gemm        projections that share an input, separate vs ONE concatenated GEMM (+ LoRA A rows), forward and dgrad
  lora        PEFT-style LoRA linear (base + B(A x) * s, eager autograd) vs a fused autograd function (one GEMM over
              [W; A], addmm with s*B, dx in one GEMM over [dy | dxa]); fwd+bwd time and max |diff|
  dequant     block-FP8 (128x128 scales) -> bf16 of one MLP weight: torch ops vs a Triton kernel (FP8 storage with
              exact bf16 compute frees half the frozen-weight memory)
  gdn         fla chunk_gated_delta_rule: GVA (16 k heads, 48 v heads, native) vs repeat_interleave'd q/k, varlen
  conv        GDN short conv: HF's depthwise F.conv1d fallback vs fla's Triton causal_conv1d (varlen-aware)
  attn        16-layer attention: SDPA (mem-efficient; cuDNN SDPA is off on GH200) padded vs flash_attn varlen
  norm        HF Qwen3.5 RMSNorm / SwiGLU vs Liger kernels, fwd+bwd

Usage: python gh200_kernels.py --out result.json [--tokens 3072 6144] [--rows 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _c in filter(None, os.environ.get("PERF_KIT_EXTRA_PATHS", "").split(os.pathsep)):   # extra library dirs
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

import torch
import torch.nn.functional as F

DEV = "cuda"
BF = torch.bfloat16
H, I, R = 5120, 17408, 128          # hidden, intermediate, LoRA rank
NK, NV, DK, DV = 16, 48, 128, 128   # GDN key / value heads and dims
NQ, NKV, HD = 24, 4, 256            # attention heads, kv heads, head dim


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def case_launch(res, args):
    x = torch.zeros(16, device=DEV)
    n = 5000

    def eager():
        for _ in range(n):
            x.add_(1.0)
    t = timeit(eager, iters=3, warmup=1)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(500):
            x.add_(1.0)
    tg = timeit(lambda: g.replay(), iters=10, warmup=2)
    res["launch"] = {"eager_us_per_kernel": round(t * 1000 / n, 2), "graph_us_per_kernel": round(tg * 1000 / 500, 3)}


def tflops(flop, ms):
    return round(flop / (ms * 1e-3) / 1e12, 1)


def case_gemm(res, args):
    groups = {   # name: (in_features, [out features of the projections sharing the input])
        "gdn_in": (H, [2 * NK * DK + NV * DV, NV * DV, NV, NV]),
        "attn_in": (H, [2 * NQ * HD, NKV * HD, NKV * HD]),
        "mlp_in": (H, [I, I]),
        "gdn_out": (NV * DV, [H]),
        "attn_out": (NQ * HD, [H]),
        "mlp_down": (I, [H]),
    }
    out = {}
    for T in args.tokens:
        for name, (k, outs) in groups.items():
            x = torch.randn(T, k, device=DEV, dtype=BF)
            ws = [torch.randn(o, k, device=DEV, dtype=BF) * 0.02 for o in outs]
            n_lora = sum(1 for o in outs if o >= 1024)     # LoRA on the wide projections only
            wcat = torch.cat(ws + [torch.randn(R * n_lora, k, device=DEV, dtype=BF) * 0.02])
            dys = [torch.randn(T, o, device=DEV, dtype=BF) for o in outs]
            dycat = torch.randn(T, wcat.shape[0], device=DEV, dtype=BF)
            f_sep = 2 * T * k * sum(outs)
            f_cat = 2 * T * k * wcat.shape[0]
            t_sep = timeit(lambda: [x @ w.t() for w in ws])
            t_cat = timeit(lambda: x @ wcat.t())
            d_sep = timeit(lambda: [dy @ w for dy, w in zip(dys, ws)])
            d_cat = timeit(lambda: dycat @ wcat)
            out[f"{name}_T{T}"] = {"fwd_sep_ms": round(t_sep, 3), "fwd_sep_tf": tflops(f_sep, t_sep),
                                   "fwd_cat_ms": round(t_cat, 3), "fwd_cat_tf": tflops(f_cat, t_cat),
                                   "dgrad_sep_ms": round(d_sep, 3), "dgrad_sep_tf": tflops(f_sep, d_sep),
                                   "dgrad_cat_ms": round(d_cat, 3), "dgrad_cat_tf": tflops(f_cat, d_cat)}
            del x, ws, wcat, dys, dycat
    res["gemm"] = out


class FusedLoRA(torch.autograd.Function):
    """y = x W^T + (x A^T) (s B)^T with W frozen, wcat = [W; A] (bf16), sb = s * B (bf16). Saves x and xa only."""

    @staticmethod
    def forward(ctx, x, wcat, a, sb, n_out):            # a: the trainable A whose bf16 values are wcat[n_out:]
        h = x @ wcat.t()
        base, xa = h[:, :n_out], h[:, n_out:]
        y = torch.addmm(base, xa, sb.t())
        ctx.save_for_backward(x, xa, wcat, sb)
        ctx.n_out = n_out
        return y

    @staticmethod
    def backward(ctx, dy):
        x, xa, wcat, sb = ctx.saved_tensors
        dsb = dy.t() @ xa                                  # [out, r]
        dxa = dy @ sb                                      # [T, r]
        dx = torch.cat([dy, dxa], 1) @ wcat                # dy W + dxa A in one GEMM
        dA = dxa.t() @ x                                   # [r, in]
        return dx, None, dA, dsb, None


def case_lora(res, args):
    out = {}
    T = max(args.tokens)
    for name, (k, n) in {"mlp_up": (H, I), "mlp_down": (I, H), "gdn_qkv": (H, 2 * NK * DK + NV * DV)}.items():
        x0 = torch.randn(T, k, device=DEV, dtype=BF)
        W = torch.randn(n, k, device=DEV, dtype=BF) * 0.02
        A = (torch.randn(R, k, device=DEV) * 0.02).to(BF).requires_grad_(True)
        B = (torch.randn(n, R, device=DEV) * 0.02).to(BF).requires_grad_(True)
        s = 2.0
        dy = torch.randn(T, n, device=DEV, dtype=BF)

        def peft_like():
            x = x0.detach().requires_grad_(True)
            y = x @ W.t() + (x @ A.t() @ B.t()) * s
            y.backward(dy)
            return x.grad
        wcat = torch.cat([W, A.detach()])
        sbp = (B.detach() * s).requires_grad_(True)
        ap = A.detach().clone().requires_grad_(True)

        def fused():
            x = x0.detach().requires_grad_(True)
            y = FusedLoRA.apply(x, wcat, ap, sbp, n)
            y.backward(dy)
            return x.grad
        t_p = timeit(peft_like, iters=10, warmup=3)
        t_f = timeit(fused, iters=10, warmup=3)
        A.grad = B.grad = sbp.grad = ap.grad = None
        g1 = peft_like()
        g2 = fused()
        out[name] = {"peft_like_ms": round(t_p, 3), "fused_ms": round(t_f, 3),
                     "dx_max_abs_diff": float((g1.float() - g2.float()).abs().max()),
                     "dx_max_abs": float(g1.float().abs().max())}
    res["lora"] = out


def case_dequant(res, args):
    import triton
    import triton.language as tl

    @triton.jit
    def _deq(w_ptr, s_ptr, o_ptr, N, K, KB, BLOCK: tl.constexpr):
        pid_n, pid_k = tl.program_id(0), tl.program_id(1)
        rn = pid_n * BLOCK + tl.arange(0, BLOCK)
        rk = pid_k * BLOCK + tl.arange(0, BLOCK)
        m = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=m, other=0.0).to(tl.float32)
        sc = tl.load(s_ptr + pid_n * KB + pid_k)
        tl.store(o_ptr + rn[:, None] * K + rk[None, :], (w * sc).to(tl.bfloat16), mask=m)

    n, k = I, H
    w = (torch.randn(n, k, device=DEV) * 0.02).to(torch.float8_e4m3fn)
    sc = torch.rand((n + 127) // 128, (k + 127) // 128, device=DEV) + 0.5

    def ref():
        s = sc.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        return (w.float() * s).to(BF)
    o = torch.empty(n, k, device=DEV, dtype=BF)

    def tri():
        _deq[(triton.cdiv(n, 128), triton.cdiv(k, 128))](w, sc, o, n, k, sc.shape[1], BLOCK=128)
        return o
    t_r = timeit(ref)
    t_t = timeit(tri)
    exact = bool(torch.equal(ref(), tri()))
    gb = (n * k * 1 + n * k * 2) / 1e9
    res["dequant"] = {"shape": [n, k], "torch_ms": round(t_r, 3), "triton_ms": round(t_t, 3),
                      "triton_gbps": round(gb / (t_t * 1e-3), 1), "bit_exact": exact,
                      "per_layer_ms_est": round(t_t * (3 * I * H + 2.4 * H * 6144) / (I * H), 3)}


def cu_seqlens_for(T, rows):
    lens = [T // rows] * rows
    lens[-1] += T - sum(lens)
    c = torch.zeros(rows + 1, dtype=torch.int32)
    c[1:] = torch.tensor(lens).cumsum(0)
    return c.to(DEV), lens


def case_gdn(res, args):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    out = {}
    for T in args.tokens:
        cu, _ = cu_seqlens_for(T, args.rows)
        cu = cu.long()
        q = torch.randn(1, T, NK, DK, device=DEV, dtype=BF)
        k = torch.randn(1, T, NK, DK, device=DEV, dtype=BF)
        v = torch.randn(1, T, NV, DV, device=DEV, dtype=BF)
        g = -torch.rand(1, T, NV, device=DEV, dtype=torch.float32) * 0.1
        beta = torch.rand(1, T, NV, device=DEV, dtype=BF)
        do = torch.randn(1, T, NV, DV, device=DEV, dtype=BF)

        def run(gva: bool):
            qq, kk, vv = (t.detach().requires_grad_(True) for t in (q, k, v))
            if gva:
                qi, ki = qq, kk
            else:
                qi, ki = qq.repeat_interleave(NV // NK, 2), kk.repeat_interleave(NV // NK, 2)
            o, _ = chunk_gated_delta_rule(qi, ki, vv, g=g, beta=beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
            o.backward(do)
            return o, qq.grad
        rec = {}
        for gva in (False, True):
            try:
                rec[f"{'gva' if gva else 'repeat'}_ms"] = round(timeit(lambda: run(gva), iters=10, warmup=3), 3)
            except Exception as ex:   # noqa: BLE001
                rec[f"{'gva' if gva else 'repeat'}_error"] = repr(ex)[:300]
        try:
            o1, g1 = run(False)
            o2, g2 = run(True)
            rec["out_max_diff"] = float((o1.float() - o2.float()).abs().max())
            rec["dq_max_diff"] = float((g1.float() - g2.float()).abs().max())
        except Exception as ex:       # noqa: BLE001
            rec["diff_error"] = repr(ex)[:300]
        out[f"T{T}"] = rec
    res["gdn"] = out


def case_conv(res, args):
    from fla.modules.conv import causal_conv1d
    out = {}
    D = 2 * NK * DK + NV * DV
    for T in args.tokens:
        x0 = torch.randn(1, T, D, device=DEV, dtype=BF)
        w = torch.randn(D, 4, device=DEV, dtype=BF) * 0.3
        dy = torch.randn(1, T, D, device=DEV, dtype=BF)
        cu, _ = cu_seqlens_for(T, args.rows)

        def hf():
            x = x0.detach().requires_grad_(True)
            y = F.silu(F.conv1d(x.transpose(1, 2), w.unsqueeze(1), padding=3, groups=D)[:, :, :T]).transpose(1, 2)
            y.backward(dy)
            return y, x.grad

        def fl(cus=None):
            x = x0.detach().requires_grad_(True)
            y, _ = causal_conv1d(x, w, activation="silu", cu_seqlens=cus)
            y.backward(dy)
            return y, x.grad
        rec = {"hf_ms": round(timeit(hf, iters=10, warmup=3), 3)}
        try:
            rec["fla_ms"] = round(timeit(fl, iters=10, warmup=3), 3)
            rec["fla_varlen_ms"] = round(timeit(lambda: fl(cu.long()), iters=10, warmup=3), 3)
            y1, g1 = hf()
            y2, g2 = fl()
            rec["y_max_diff"] = float((y1.float() - y2.float()).abs().max())
            rec["dx_max_diff"] = float((g1.float() - g2.float()).abs().max())
        except Exception as ex:   # noqa: BLE001
            rec["fla_error"] = repr(ex)[:300]
        out[f"T{T}"] = rec
    res["conv"] = out


def case_attn(res, args):
    out = {}
    torch.backends.cuda.enable_cudnn_sdp(False)
    for T in args.tokens:
        cu, lens = cu_seqlens_for(T, args.rows)
        L = max(lens)
        q = torch.randn(args.rows, NQ, L, HD, device=DEV, dtype=BF)
        k = torch.randn(args.rows, NKV, L, HD, device=DEV, dtype=BF)
        v = torch.randn(args.rows, NKV, L, HD, device=DEV, dtype=BF)
        do = torch.randn(args.rows, NQ, L, HD, device=DEV, dtype=BF)

        def sdpa():
            qq, kk, vv = (t.detach().requires_grad_(True) for t in (q, k, v))
            o = F.scaled_dot_product_attention(qq, kk.repeat_interleave(NQ // NKV, 1), vv.repeat_interleave(NQ // NKV, 1),
                                               is_causal=True)
            o.backward(do)
        rec = {"sdpa_padded_ms": round(timeit(sdpa, iters=10, warmup=3), 3)}
        try:
            from flash_attn import flash_attn_varlen_func
            qv = torch.randn(T, NQ, HD, device=DEV, dtype=BF)
            kv = torch.randn(T, NKV, HD, device=DEV, dtype=BF)
            vv0 = torch.randn(T, NKV, HD, device=DEV, dtype=BF)
            dov = torch.randn(T, NQ, HD, device=DEV, dtype=BF)

            def fa():
                qq, kk, vv = (t.detach().requires_grad_(True) for t in (qv, kv, vv0))
                o = flash_attn_varlen_func(qq, kk, vv, cu, cu, L, L, causal=True)
                o.backward(dov)
            rec["flash_varlen_ms"] = round(timeit(fa, iters=10, warmup=3), 3)
        except Exception as ex:   # noqa: BLE001
            rec["flash_error"] = repr(ex)[:300]
        out[f"T{T}"] = rec
    res["attn"] = out


def case_norm(res, args):
    out = {}
    T = max(args.tokens)
    x0 = torch.randn(T, H, device=DEV, dtype=BF)
    wn = torch.zeros(H, device=DEV, dtype=BF).requires_grad_(True)
    dy = torch.randn(T, H, device=DEV, dtype=BF)

    def hf_norm():
        x = x0.detach().requires_grad_(True)
        o = x.float()
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + wn.float())
        o.type_as(x).backward(dy)
        return o.type_as(x)
    out["rmsnorm_hf_ms"] = round(timeit(hf_norm), 3)
    g0 = torch.randn(T, I, device=DEV, dtype=BF)
    u0 = torch.randn(T, I, device=DEV, dtype=BF)
    dyi = torch.randn(T, I, device=DEV, dtype=BF)

    def hf_swiglu():
        g, u = g0.detach().requires_grad_(True), u0.detach().requires_grad_(True)
        (F.silu(g) * u).backward(dyi)
    out["swiglu_hf_ms"] = round(timeit(hf_swiglu), 3)
    try:
        from liger_kernel.transformers.rms_norm import LigerRMSNormForQwen3Next
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        ln = LigerRMSNormForQwen3Next(H, eps=1e-6).to(DEV, BF)

        def lg_norm():
            x = x0.detach().requires_grad_(True)
            o = ln(x)
            o.backward(dy)
            return o
        out["rmsnorm_liger_ms"] = round(timeit(lg_norm), 3)
        with torch.no_grad():
            ln.weight.zero_()
        out["rmsnorm_max_diff"] = float((hf_norm().float() - lg_norm().float()).abs().max())

        def lg_swiglu():
            g, u = g0.detach().requires_grad_(True), u0.detach().requires_grad_(True)
            LigerSiLUMulFunction.apply(g, u).backward(dyi)
        out["swiglu_liger_ms"] = round(timeit(lg_swiglu), 3)
    except Exception as ex:   # noqa: BLE001
        out["liger_error"] = repr(ex)[:300]
    res["norm"] = out


CASES = {"launch": case_launch, "gemm": case_gemm, "lora": case_lora, "dequant": case_dequant, "gdn": case_gdn,
         "conv": case_conv, "attn": case_attn, "norm": case_norm}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[3072, 6144])
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    args = ap.parse_args(argv)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "time": time.strftime("%F %T")}
    for name in args.cases:
        t0 = time.time()
        try:
            CASES[name](res, args)
        except Exception as ex:   # noqa: BLE001  (one failing case must not stop the others)
            res[name] = {"error": repr(ex)[:400], "trace": traceback.format_exc()[-1200:]}
        torch.cuda.empty_cache()
        print(json.dumps({name: res.get(name), "secs": round(time.time() - t0, 1)}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
