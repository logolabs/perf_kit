"""GPU checks of perf_kit's fast Qwen3.5 path before any real run (one GPU, a few minutes):

  equiv     a 4-layer Qwen3.5 at the 27B's head shapes (GDN 128-d heads with 3x GVA, 256-d attention heads), bf16:
            fast kernels vs transformers + PEFT (padded batch, sdpa, fla) on the same packed rows -- per-target CE
            and LoRA-gradient agreement (bf16 tolerances), and the reference kernels in fp32 as a tie-breaker
  fp8       the same model with block-FP8 frozen weights: FP8 storage (exact dequant per GEMM) must give
            bit-identical CE to bf16 storage of the dequantised weights
  layer     one GDN and one attention layer at the 27B's full width, fwd+bwd on 8 packed rows of T tokens:
            fast vs reference kernels, ms
  dequant   Triton block-FP8 -> bf16 vs torch, bit-exact, GB/s

Usage: python gpu_equiv.py --out result.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _c in [*filter(None, os.environ.get("PERF_KIT_EXTRA_PATHS", "").split(os.pathsep)),
           str(Path(__file__).resolve().parent.parent)]:   # extra library dirs (e.g. --target installs), then the repo
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c)

import torch

TARGETS = ["in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj"]


def cfg_small(hidden=1280, inter=4352, layers=4, vocab=4096):
    from transformers import Qwen3_5TextConfig
    c = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                          num_attention_heads=6, num_key_value_heads=1, head_dim=256, linear_num_key_heads=4,
                          linear_num_value_heads=12, linear_key_head_dim=128, linear_value_head_dim=128,
                          linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=False,
                          max_position_embeddings=8192,
                          layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
                          rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                                           "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10],
                                           "mrope_interleaved": True})
    c._attn_implementation = "sdpa"
    return c


def rows(n, lo, hi, vocab, seed=0):
    from perf_kit.packing import IGNORE
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        L = int(torch.randint(lo, hi, (1,), generator=g))
        ids = torch.randint(0, vocab, (L,), generator=g)
        p = L // 3
        lab = ids.clone()
        lab[:p] = IGNORE
        seg = torch.zeros(L, dtype=torch.long)
        seg[p:] = 2
        out.append((ids.tolist(), lab.tolist(), seg.tolist()))
    return out


def hf_ce(pm, rs, dev):
    from perf_kit.packing import IGNORE
    n = max(len(r[0]) for r in rs)
    ids = torch.zeros(len(rs), n, dtype=torch.long)
    lab = torch.full((len(rs), n), IGNORE, dtype=torch.long)
    att = torch.zeros(len(rs), n, dtype=torch.long)
    for i, (a, b, _) in enumerate(rs):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
    logits = pm(input_ids=ids, attention_mask=att).logits
    tgt, lg = lab[:, 1:], logits[:, :-1].float()
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE,
                                           reduction="none").view(tgt.shape)
    return torch.cat([ce[i, :len(r[0]) - 1][tgt[i, :len(r[0]) - 1] != IGNORE] for i, r in enumerate(rs)])


def build_pair(cfg, dev, seed=0, fp8=False):
    import peft
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    torch.manual_seed(seed)
    hf = Qwen3_5ForCausalLM(cfg).to(dev, torch.bfloat16)
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n or "dt_bias" in n:
                p.add_(torch.randn_like(p) * 0.1)
    sd = {k: v.detach().clone() for k, v in hf.state_dict().items()}
    sd_fp8 = None
    if fp8:                                         # block-quantise every big Linear, as the FP8 checkpoint stores it
        from perf_kit.kernels.dequant import dequant_torch
        sd_fp8 = {}
        for k, v in sd.items():
            if k.endswith("proj.weight") and "in_proj_a" not in k and "in_proj_b" not in k and v.dim() == 2:
                n_, k_ = v.shape
                s = torch.zeros((n_ + 127) // 128, (k_ + 127) // 128, device=dev)
                for i in range(s.shape[0]):
                    for j in range(s.shape[1]):
                        s[i, j] = v[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128].abs().max().float() / 448.0
                sexp = s.repeat_interleave(128, 0)[:n_].repeat_interleave(128, 1)[:, :k_]
                codes = (v.float() / sexp).to(torch.float8_e4m3fn)
                sd_fp8[k] = codes
                sd_fp8[k + "_scale_inv"] = s.to(torch.bfloat16).float()
                sd[k] = dequant_torch(codes, sd_fp8[k + "_scale_inv"], torch.bfloat16)
            else:
                sd_fp8[k] = v
        hf.load_state_dict(sd, strict=False)
    pm = peft.get_peft_model(hf, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                                  bias="none"))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.02)
    lora = {k.replace(".default", ""): v.detach().float().cpu() for k, v in pm.state_dict().items() if "lora_" in k}
    return pm, sd, sd_fp8, lora


def fast(cfg, sd, lora, dev, impl="fast", dtype=torch.bfloat16, storage="bf16"):
    from perf_kit.models.qwen3_5 import FastQwen35
    from perf_kit.weights import DictSource
    m = FastQwen35(cfg, DictSource(sd), prefix="model.", rank=16, alpha=32, storage=storage, device=dev, dtype=dtype,
                   impl=impl)
    m.lora.load_peft(lora)
    return m


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / (b.norm() + 1e-12))


def case_equiv(res, dev):
    from perf_kit.packing import pack
    cfg = cfg_small()
    pm, sd, _, lora = build_pair(cfg, dev)
    rs = rows(8, 200, 900, cfg.vocab_size)
    pm.train()
    ref = hf_ce(pm, rs, dev)
    ref.sum().backward()
    grads = {n.replace(".default", ""): p.grad.float() for n, p in pm.named_parameters() if "lora_" in n}
    out = {}
    for impl, dt in (("fast", torch.bfloat16), ("ref", torch.bfloat16), ("ref", torch.float32)):
        sdx = {k: v.to(dt) for k, v in sd.items()} if dt != torch.bfloat16 else sd
        m = fast(cfg, sdx, lora, dev, impl=impl, dtype=dt)
        ce = m.token_ce(pack(rs).to(dev))
        ce.sum().backward()
        ga = [rel(mm.a_grad, grads[f"base_model.model.{n}.lora_A.weight"]) for n, mm in m.lora.slots.items()]
        gb = [rel(mm.b_grad, grads[f"base_model.model.{n}.lora_B.weight"]) for n, mm in m.lora.slots.items()]
        out[f"{impl}_{str(dt)[6:]}"] = {"ce_rel": rel(ce.detach(), ref.detach()),
                                        "ce_mean": float(ce.mean()), "hf_ce_mean": float(ref.mean()),
                                        "gradA_rel_max": max(ga), "gradA_rel_mean": sum(ga) / len(ga),
                                        "gradB_rel_max": max(gb), "gradB_rel_mean": sum(gb) / len(gb)}
        del m
    res["equiv"] = out


def case_fp8(res, dev):
    from perf_kit.packing import pack
    cfg = cfg_small()
    _, sd, sd_fp8, lora = build_pair(cfg, dev, seed=1, fp8=True)
    rs = rows(8, 200, 900, cfg.vocab_size, seed=1)
    p = pack(rs).to(dev)
    a = fast(cfg, sd, lora, dev, storage="bf16")
    ca = a.token_ce(p)
    ca.sum().backward()
    ga = a.lora.grad.clone()
    del a
    a2 = fast(cfg, sd, lora, dev, storage="bf16")          # control: the same bf16 run again
    a2.token_ce(p).sum().backward()
    ga2 = a2.lora.grad.clone()
    del a2
    b = fast(cfg, sd_fp8, lora, dev, storage="fp8")
    cb = b.token_ce(p)
    cb.sum().backward()
    gb = b.lora.grad

    def grel(x, y):
        return float((x - y).norm() / y.norm().clamp_min(1e-30))
    # Some backward kernels (fla's Triton GDN backward) accumulate with atomics, so gradients are not bitwise
    # reproducible even between two identical runs; compare fp8-vs-bf16 against that run-to-run floor.
    res["fp8"] = {"ce_bit_equal": bool(torch.equal(ca, cb)), "grad_bit_equal": bool(torch.equal(ga, gb)),
                  "grad_bit_equal_bf16_rerun": bool(torch.equal(ga, ga2)), "grad_rel_fp8_vs_bf16": grel(gb, ga),
                  "grad_rel_bf16_rerun": grel(ga2, ga), "ce_max_diff": float((ca - cb).abs().max()),
                  "stored": b.stored_bytes()}


def case_layer(res, dev, tokens):
    from perf_kit.packing import pack
    cfg = cfg_small(hidden=5120, inter=17408, layers=4, vocab=4096)
    cfg.linear_num_key_heads, cfg.linear_num_value_heads = 16, 48
    cfg.num_attention_heads, cfg.num_key_value_heads = 24, 4
    torch.manual_seed(0)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    with torch.device(dev):
        hf = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    sd = {k: v.detach() for k, v in hf.state_dict().items()}
    del hf
    lora = None
    out = {}
    for impl in ("fast", "ref"):
        from perf_kit.models.qwen3_5 import FastQwen35
        from perf_kit.weights import DictSource
        m = FastQwen35(cfg, DictSource(sd), prefix="model.", rank=128, alpha=256, storage="bf16", device=dev,
                       impl=impl)
        m.lora.init_new(0)
        for T in tokens:
            rs = rows(8, T // 8, T // 8 + 1, cfg.vocab_size, seed=2)
            p = pack(rs).to(dev)

            def run():
                m.token_ce(p).sum().backward()
            for _ in range(2):
                run()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                run()
            torch.cuda.synchronize()
            out[f"{impl}_T{T}_ms_per_4_layers"] = round((time.time() - t0) / 3 * 1000, 1)
        out[f"{impl}_peak_gb"] = round(torch.cuda.max_memory_allocated(dev) / 2**30, 1)
        del m
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    res["layer"] = out


def case_dequant(res, dev):
    from perf_kit.kernels.dequant import dequant_into, dequant_torch
    n, k = 17408, 5120
    w = (torch.randn(n, k, device=dev) * 0.02).to(torch.float8_e4m3fn)
    sc = (torch.rand((n + 127) // 128, (k + 127) // 128, device=dev) + 0.5).to(torch.bfloat16)
    o = torch.empty(n, k, device=dev, dtype=torch.bfloat16)
    ref = dequant_torch(w, sc)
    dequant_into(o, w, sc.float())
    for _ in range(3):
        dequant_into(o, w, sc.float())
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        dequant_into(o, w, sc.float())
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    res["dequant"] = {"bit_exact": bool(torch.equal(o, ref)), "ms": round(ms, 3),
                      "gbps": round(n * k * 3 / 1e9 / (ms / 1000), 1)}


def case_fp8fwd(res, dev, tokens):
    """Forward GEMMs on the checkpoint's own block-FP8 weights (vLLM's CUTLASS block-scaled kernel, activations
    quantised per 1x128 group) vs dequant -> bf16 cuBLAS, at the 27B's run shapes."""
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
    from perf_kit.kernels.dequant import dequant_into
    shapes = {"gdn_in": (16384, 5120), "attn_in": (14336, 5120), "mlp_in": (34816, 5120), "gdn_out": (5120, 6144),
              "mlp_down": (5120, 17408)}
    out = {}
    for T in tokens:
        for name, (n, k) in shapes.items():
            w = (torch.randn(n, k, device=dev) * 0.02).to(torch.float8_e4m3fn)
            sc = (torch.rand(n // 128, k // 128, device=dev) * 0.01 + 0.005)
            x = torch.randn(T, k, device=dev, dtype=torch.bfloat16)
            wb = torch.empty(n, k, device=dev, dtype=torch.bfloat16)

            def ref():
                dequant_into(wb, w, sc)
                return x @ wb.t()

            def gemm_only():
                return x @ wb.t()

            def fp8():
                xq, xs = per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=False)
                return vops.cutlass_scaled_mm(xq, w.t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=sc.t())
            try:
                y_ref, y8 = ref(), fp8()
                t_ref = timeit_cuda(ref)
                t_g = timeit_cuda(gemm_only)
                t_8 = timeit_cuda(fp8)
                out[f"{name}_T{T}"] = {"dequant_bf16_ms": round(t_ref, 3), "bf16_gemm_ms": round(t_g, 3),
                                       "fp8_ms": round(t_8, 3), "speedup_vs_dequant_bf16": round(t_ref / t_8, 3),
                                       "rel_err": rel(y8, y_ref)}
            except Exception as ex:          # noqa: BLE001
                out[f"{name}_T{T}"] = {"error": repr(ex)[:300]}
    res["fp8fwd"] = out


def case_dense_equiv(res, dev):
    """FastDense (fast kernels: Liger llama-mode RMSNorm + fused add-norm, flash varlen, SwiGLU) vs transformers +
    PEFT, bf16, random small Llama-3 (llama3 rope scaling), Qwen2 (q/k/v bias) and Qwen3 (q/k norm) models."""
    import peft
    import transformers
    from perf_kit.models.dense import FastDense, DEFAULT_TARGETS
    from perf_kit.packing import pack
    from perf_kit.weights import DictSource
    common = dict(vocab_size=4096, hidden_size=1024, intermediate_size=2816, num_hidden_layers=4, num_attention_heads=8,
                  num_key_value_heads=2, rms_norm_eps=1e-5, tie_word_embeddings=False, max_position_embeddings=8192)
    out = {}
    for kind in ("llama", "qwen2", "qwen3"):
        if kind == "llama":
            cfg = transformers.LlamaConfig(**common, head_dim=128, rope_theta=500000.0,
                                           rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                         "high_freq_factor": 4.0,
                                                         "original_max_position_embeddings": 1024})
            cls = transformers.LlamaForCausalLM
        elif kind == "qwen2":
            cfg = transformers.Qwen2Config(**common, rope_theta=1000000.0)
            cls = transformers.Qwen2ForCausalLM
        else:
            cfg = transformers.Qwen3Config(**common, head_dim=128, rope_theta=1000000.0)
            cls = transformers.Qwen3ForCausalLM
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(0)
        hf = cls(cfg).to(dev, torch.bfloat16)
        with torch.no_grad():
            for n, p_ in hf.named_parameters():
                if "norm" in n or n.endswith("bias"):
                    p_.add_(torch.randn_like(p_) * 0.1)
        sd = {k: v.detach().clone() for k, v in hf.state_dict().items()}
        pm = peft.get_peft_model(hf, peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                                      target_modules=list(DEFAULT_TARGETS)))
        with torch.no_grad():
            for n, p_ in pm.named_parameters():
                if "lora_B" in n:
                    p_.copy_(torch.randn_like(p_) * 0.02)
        lora = {k.replace(".default", ""): v.detach().float().cpu() for k, v in pm.state_dict().items() if "lora_" in k}
        rs = rows(8, 200, 900, cfg.vocab_size, seed=3)
        pm.train()
        ref = hf_ce(pm, rs, dev)
        ref.sum().backward()
        grads = {n.replace(".default", ""): p_.grad.float() for n, p_ in pm.named_parameters() if "lora_" in n}
        res_k, keep = {}, {}
        for impl, dt in (("fast", torch.bfloat16), ("ref", torch.bfloat16), ("ref", torch.float32)):
            sdx = {k: v.to(dt) for k, v in sd.items()} if dt != torch.bfloat16 else sd
            m = FastDense(cfg, DictSource(sdx), prefix="model.", rank=16, alpha=32, storage="bf16", device=dev,
                          impl=impl, dtype=dt)
            m.lora.load_peft(lora)
            ce = m.token_ce(pack(rs).to(dev))
            ce.sum().backward()
            ga = [rel(mm.a_grad, grads[f"base_model.model.{n}.lora_A.weight"]) for n, mm in m.lora.slots.items()]
            gb = [rel(mm.b_grad, grads[f"base_model.model.{n}.lora_B.weight"]) for n, mm in m.lora.slots.items()]
            tag = f"{impl}_{str(dt)[6:]}"
            res_k[tag] = {"ce_rel": rel(ce.detach(), ref.detach()), "gradA_rel_max": max(ga), "gradB_rel_max": max(gb),
                          "gradA_rel_mean": sum(ga) / len(ga)}
            keep[tag] = (ce.detach().float(), {n: mm.a_grad.clone() for n, mm in m.lora.slots.items()})
            del m
        ce_t, g_t = keep["ref_float32"]                      # the closest thing to the exact answer
        ce_f, g_f = keep["fast_bfloat16"]
        res_k["fast_vs_fp32"] = {"ce_rel": rel(ce_f, ce_t), "gradA_rel_max": max(rel(g_f[n], g_t[n]) for n in g_t)}
        res_k["hf_vs_fp32"] = {"ce_rel": rel(ref.detach().float(), ce_t),
                               "gradA_rel_max": max(rel(grads[f"base_model.model.{n}.lora_A.weight"], g_t[n]) for n in g_t)}
        out[kind] = res_k
        del pm, hf
        torch.cuda.empty_cache()
    res["dense_equiv"] = out


def timeit_cuda(fn, iters=20, warmup=5):
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", nargs="+", default=["dequant", "equiv", "fp8", "layer"])
    ap.add_argument("--tokens", type=int, nargs="+", default=[3072, 6144])
    a = ap.parse_args(argv)
    dev = torch.device("cuda", 0)
    torch.backends.cuda.enable_cudnn_sdp(False)
    from perf_kit.kernels import ops
    res = {"torch": torch.__version__, "kernels": ops.available()}
    for c in a.cases:
        t0 = time.time()
        try:
            if c == "fp8fwd":
                case_fp8fwd(res, dev, a.tokens)
            elif c == "dense_equiv":
                case_dense_equiv(res, dev)
            else:
                {"equiv": case_equiv, "fp8": case_fp8, "dequant": case_dequant}.get(
                    c, lambda r, d: case_layer(r, d, a.tokens))(res, dev)
        except Exception as ex:                     # noqa: BLE001
            res[c] = {"error": repr(ex)[:500], "trace": traceback.format_exc()[-2000:]}
        torch.cuda.empty_cache()
        print(json.dumps({c: res.get(c), "secs": round(time.time() - t0, 1)}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
