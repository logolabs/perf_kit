"""GPU checks (skipped without CUDA): the start-up activation measurement predicts what a real forward + backward
stores, and the opt-in kernel backends (FA3 attention, FlashQLA Gated DeltaNet) agree with an fp32 reference as
closely as the default kernels do. The big fast-vs-transformers equivalence check is bench/gpu_equiv.py."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from perf_kit.kernels import ops                      # noqa: E402
from perf_kit.packing import IGNORE, pack             # noqa: E402

eq = pytest.importorskip("test_qwen3_5_equiv")


def _rows(n, L, vocab=101, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ids = torch.randint(0, vocab, (L,), generator=g).tolist()
        out.append((ids, [IGNORE] * 4 + ids[4:], [2] * L))
    return out


def _fast_bf16():
    from perf_kit.models.qwen3_5 import FastQwen35
    from perf_kit.weights import DictSource
    cfg, hf, pm = eq.build()
    sd = {k.replace("base_layer.", ""): v.detach().float() for k, v in hf.state_dict().items() if "lora_" not in k}
    fm = FastQwen35(cfg, DictSource(sd), prefix="model.", head_name="lm_head.weight", rank=4, alpha=8, storage="bf16",
                    device="cuda", dtype=torch.bfloat16)
    fm.lora.init_new(0)
    return fm


def test_measured_activations_match_a_real_step():
    from perf_kit.engine import measure_act_mb
    fm = _fast_bf16()
    small = measure_act_mb(fm, pack(_rows(4, 128)).to("cuda"))
    big = measure_act_mb(fm, pack(_rows(8, 128, seed=1)).to("cuda"))
    assert small["method"] == "cuda_allocator" and 0 < small["in_mb"] < small["act_mb"]
    # per token: ~linear in tokens (the tiny model has fixed per-forward overheads, e.g. RoPE tables, of ~10-15%)
    assert abs(small["act_mb"] - big["act_mb"]) / big["act_mb"] < 0.3
    p = pack(_rows(8, 128, seed=2)).to("cuda")
    T, n = int(p.ids.numel()), len(fm.layers)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    h = fm.hidden(p, None)
    torch.cuda.synchronize()
    stored = torch.cuda.memory_allocated() - m0
    pred = big["act_mb"] * T * n * 2**20
    assert 0.8 * stored <= pred <= 1.2 * stored, (pred, stored)
    h.float().sum().backward()


def _ref_attention(q, k, v, packed, scale):
    return ops.attention_ref(q.float(), k.float(), v.float(), packed, scale)


@pytest.mark.skipif(ops._lib("fa3") is None or ops._lib("flash") is None, reason="needs flash-attn 2 and 3")
def test_fa3_matches_fa2_against_fp32(monkeypatch):
    torch.manual_seed(0)
    p = pack(_rows(3, 300) + _rows(2, 77, seed=5)).to("cuda")
    T = int(p.ids.numel())
    q = torch.randn(T, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, 2, 128, device="cuda", dtype=torch.bfloat16)
    ref = _ref_attention(q, k, v, p, 128 ** -0.5)
    errs = {}
    for be in ("fa2", "fa3"):
        monkeypatch.setattr(ops, "ATTN_BACKEND", be)
        o = ops.attention(q, k, v, p, 128 ** -0.5).float()
        errs[be] = ((o - ref).norm() / ref.norm()).item()
    assert errs["fa3"] < 1.5 * errs["fa2"] + 1e-3, errs


@pytest.mark.skipif(ops._lib("flashqla") is None or ops._lib("fla_gdn") is None, reason="needs fla and FlashQLA")
def test_flashqla_matches_fla_against_fp32(monkeypatch):
    torch.manual_seed(0)
    p = pack(_rows(2, 256) + _rows(1, 100, seed=3)).to("cuda")
    T, H, HV, D = int(p.ids.numel()), 2, 4, 128
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(T, HV, D, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(T, HV, device="cuda") * 0.5
    beta = torch.rand(T, HV, device="cuda", dtype=torch.bfloat16)
    pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    ref = ops.gdn_ref(q.float(), k.float(), v.float(), g, beta.float(), p).float()
    errs = {}
    for be in ("fla", "flashqla"):
        monkeypatch.setattr(ops, "GDN_BACKEND", be)
        o = ops.gdn(q, k, v, g, beta, p).float()
        errs[be] = ((o - ref).norm() / ref.norm()).item()
    assert errs["flashqla"] < 1.5 * errs["fla"] + 1e-3, errs
