"""perf_kit.dit.ideogram4 on CPU in float64: the fast engine against the upstream Ideogram4Transformer (Ideogram's
own module definition, vendored in perf_kit/dit/ideogram4/upstream) with a plain autograd LoRA on its five target
Linears. TF32 / bf16 never enter, so differences are float64 summation order only."""
from __future__ import annotations

import copy
import json
import math
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from perf_kit.dit.ideogram4 import engine as E
from perf_kit.dit.ideogram4 import train as TR
from perf_kit.dit.ideogram4.kernels import quantize_rows
from perf_kit.dit.ideogram4.stack import tiny_transformer
from perf_kit.dit.ideogram4.upstream import modeling_ideogram4 as UP
from perf_kit.dit.ideogram4.upstream.constants import (IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR,
                                                       OUTPUT_IMAGE_INDICATOR, SEQUENCE_PADDING_INDICATOR)

F64 = torch.float64


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-300))


def _sdpa_math(q, k, v, attn_mask=None, **kw):
    """Explicit-math SDPA for the reference (float64 masked SDPA kernels are not trusted on every CPU build)."""
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if attn_mask is not None:
        s = s.masked_fill(~attn_mask, float("-inf"))
    return torch.softmax(s, -1) @ v


@pytest.fixture(autouse=True)
def math_sdpa(monkeypatch):
    proxy = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    proxy.scaled_dot_product_attention = _sdpa_math
    monkeypatch.setattr(UP, "F", proxy)


class RefLora(nn.Module):
    """base(x) + s * sum_k w_k(t) B_k A_k x, autograd, the textbook definition."""

    def __init__(self, base: nn.Linear, K: int, r: int, scale: float, ctx: dict):
        super().__init__()
        self.base, self.ctx, self.scale = base, ctx, scale
        self.lora_A = nn.Parameter(torch.zeros(K, r, base.in_features, dtype=F64))
        self.lora_B = nn.Parameter(torch.zeros(K, base.out_features, r, dtype=F64))

    def forward(self, x):
        out = self.base(x)
        w = self.ctx["w"]
        xs = x.reshape(w.shape[0], -1, x.shape[-1])
        xa = torch.einsum("bli,kri->blkr", xs, self.lora_A) * w.to(x.dtype)[:, None, :, None]
        return out + self.scale * torch.einsum("blkr,kor->blo", xa, self.lora_B).reshape(out.shape)


def _models(n_bins=3, base_bins=0, rank=4, alpha=4.0, rslora=True, layers=3, seed=0):
    tr = tiny_transformer(layers=layers, dtype=F64, seed=seed)
    ref_tr, eng_tr = copy.deepcopy(tr), copy.deepcopy(tr)
    eng = E.FastDiT(eng_tr, rank=rank, alpha=alpha, rslora=rslora, n_bins=n_bins, base_bins=base_bins,
                    compute_dtype=F64, master_dtype=F64, storage="bf16", attn="ref")
    ctx: dict = {}
    for i, blk in enumerate(ref_tr.layers):
        for name in E.TARGETS:
            parent, leaf = name.split(".")
            setattr(getattr(blk, parent), leaf, RefLora(getattr(getattr(blk, parent), leaf), eng.K, rank, eng.scale,
                                                        ctx))
    ref_tr.register_forward_pre_hook(
        lambda m, a, kw: ctx.__setitem__("w", E.bin_blend_weights(kw["t"], n_bins, base_bins)), with_kwargs=True)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, kind, n in eng.lora.ref_order():
            mod = ref_tr.get_submodule(n)
            p = mod.lora_A if kind == "A" else mod.lora_B
            p.copy_(0.3 * torch.randn(p.shape, generator=g, dtype=F64))
            eng.lora.to_ref(eng.lora.master, n, kind).copy_(p)
    eng.lora.push()
    return ref_tr, eng


def build_inputs(lens, gh, gw):
    """The upstream pipeline's _build_inputs layout for given caption lengths: [pad][text][image] per row."""
    B, n_img, mt = len(lens), gh * gw, max(lens)
    L = mt + n_img
    hh = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
    ww = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
    img_pos = torch.stack([torch.zeros_like(hh), hh, ww], 1) + IMAGE_POSITION_OFFSET
    pos = torch.zeros(B, L, 3, dtype=torch.long)
    seg = torch.full((B, L), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    ind = torch.zeros(B, L, dtype=torch.long)
    for b, n in enumerate(lens):
        o = mt - n
        tp = torch.arange(n)
        pos[b, o:o + n] = torch.stack([tp, tp, tp], 1)
        pos[b, o + n:] = img_pos
        ind[b, o:o + n] = LLM_TOKEN_INDICATOR
        ind[b, o + n:] = OUTPUT_IMAGE_INDICATOR
        seg[b, o:] = 1
    return dict(position_ids=pos, segment_ids=seg, indicator=ind, mt=mt)


def _batch(eng, lens=(7, 12, 5), gh=3, gw=4, seed=2):
    B = len(lens)
    inp = build_inputs(list(lens), gh, gw)
    g = torch.Generator().manual_seed(seed)
    L = inp["mt"] + gh * gw
    llm = torch.randn(B, L, eng.cfg.llm_features_dim, generator=g, dtype=F64)
    llm = llm * (inp["indicator"] == LLM_TOKEN_INDICATOR).to(F64).unsqueeze(-1)
    x0 = torch.randn(B, gh * gw, 128, generator=g, dtype=F64)
    noise = torch.randn(B, gh * gw, 128, generator=g, dtype=F64)
    t = torch.tensor([0.05, 0.5, 0.93, 0.7][:B], dtype=torch.float32)
    tb = t.double().view(-1, 1, 1)
    z_t = (1 - tb) * noise + tb * x0
    return dict(lens=list(lens), gh=gh, gw=gw, inp=inp, llm=llm, z_t=z_t, v=x0 - noise, t=t)


def _ref_loss(ref_tr, b):
    mt = b["inp"]["mt"]
    B = b["z_t"].shape[0]
    cap = {}
    h = ref_tr.final_layer.register_forward_hook(lambda m, a, o: cap.__setitem__("o", o))
    x = torch.cat([torch.zeros(B, mt, 128, dtype=F64), b["z_t"]], 1)
    out = ref_tr(llm_features=b["llm"], x=x, t=b["t"], position_ids=b["inp"]["position_ids"],
                 segment_ids=b["inp"]["segment_ids"], indicator=b["inp"]["indicator"])
    h.remove()
    pred = cap["o"][:, mt:]
    assert pred.dtype == F64
    loss = ((pred - b["v"]) ** 2).mean(-1).mean(1).mean()
    for p in ref_tr.parameters():
        p.grad = None
    loss.backward()
    return float(loss), out.detach()


def _eng_loss(eng, b, n_save=None, loss_scale=1.0):
    p = E.pack_layout(b["lens"], b["gh"], b["gw"], "cpu")
    B = len(b["lens"])
    llm_text = b["llm"].reshape(B * p.L_pad, -1)[p.flat_text_src]
    eng.lora.zero_grad()
    loss, _ = eng.train_micro(p, llm_text, b["z_t"].reshape(B * p.n_img, -1), b["v"].reshape(B * p.n_img, -1),
                              b["t"], loss_scale, n_save=n_save)
    return float(loss)


def test_bin_weights():
    t = torch.rand(300)
    assert torch.equal(E.bin_blend_weights(t, 1, 0), torch.ones(300, 1))
    w = E.bin_blend_weights(t, 4, 0)
    assert torch.allclose(w.sum(1), torch.ones(300))
    w = E.bin_blend_weights(torch.tensor([0.99]), 4, 1)          # inside the base-only region
    assert float(w.abs().sum()) == 0.0


def test_pack_layout_matches_padded_layout():
    lens, gh, gw = [7, 12, 5], 3, 4
    inp = build_inputs(lens, gh, gw)
    p = E.pack_layout(lens, gh, gw, "cpu")
    real = (inp["segment_ids"] != SEQUENCE_PADDING_INDICATOR).reshape(-1).nonzero().squeeze(1)
    assert p.T == real.numel()
    assert torch.equal(p.pos, inp["position_ids"].reshape(-1, 3)[real])
    assert torch.equal(p.ind, (inp["indicator"].reshape(-1)[real] == OUTPUT_IMAGE_INDICATOR).long())
    assert torch.equal(real[p.text_idx], p.flat_text_src)


@pytest.mark.parametrize("n_bins,base_bins,rslora", [(1, 0, True), (3, 0, True), (4, 1, False)])
def test_loss_and_grads_match_reference(n_bins, base_bins, rslora):
    ref_tr, eng = _models(n_bins=n_bins, base_bins=base_bins, rslora=rslora)
    b = _batch(eng)
    l_ref, _ = _ref_loss(ref_tr, b)
    l_eng = _eng_loss(eng, b)
    assert abs(l_eng - l_ref) <= 1e-10 * abs(l_ref)
    nz = 0
    for _, kind, n in eng.lora.ref_order():
        mod = ref_tr.get_submodule(n)
        g_ref = (mod.lora_A if kind == "A" else mod.lora_B).grad
        got = eng.lora.to_ref(eng.lora.grad, n, kind)
        if g_ref.norm() > 0:
            nz += 1
            assert _rel(got, g_ref) < 1e-9, (n, kind, _rel(got, g_ref))
        else:
            assert float(got.abs().max()) == 0.0
    assert nz == 2 * len(eng.lora.members)                  # every adapter tensor receives gradient


def test_recompute_and_split_are_exact():
    _, eng = _models()
    b = _batch(eng, lens=(7, 12, 5, 9))
    grads = []
    for n_save in (None, 0, 1):
        _eng_loss(eng, b, n_save=n_save)
        grads.append(eng.lora.grad.clone())
    assert torch.equal(grads[0], grads[1]) and torch.equal(grads[0], grads[2])
    mt = b["inp"]["mt"]
    llm_pre = b["llm"][:, :mt]                                # the (B, mt) left-padded prefix layout
    outs = []
    for split in ("1", "2", "4", "3"):
        eng.lora.zero_grad()
        loss, T, _ = TR.train_split(eng, b["lens"], b["gh"], b["gw"], "cpu", llm_pre, b["z_t"], b["v"], b["t"], 0.5,
                                    float("inf"), split)
        outs.append((float(loss), eng.lora.grad.clone(), T))
    for l_, g, T in outs[1:]:
        assert abs(l_ - outs[0][0]) < 1e-12 * abs(outs[0][0])
        assert _rel(g, outs[0][1]) < 1e-12
        assert T == outs[0][2]
    assert _rel(outs[0][1], 0.5 * grads[0]) < 1e-12          # loss_scale scales the accumulated gradient
    assert TR.plan_split(eng, b["lens"], 12, float("inf"), "auto") == [[0, 1, 2, 3]]


def test_forward_padded_matches_reference():
    ref_tr, eng = _models(layers=2)
    b = _batch(eng)
    _, out_ref = _ref_loss(ref_tr, b)
    mt = b["inp"]["mt"]
    B = b["z_t"].shape[0]
    x = torch.cat([torch.zeros(B, mt, 128, dtype=F64), b["z_t"]], 1)
    out = E.EngineModule(eng)(llm_features=b["llm"], x=x, t=b["t"], **{k: b["inp"][k] for k in
                                                                        ("position_ids", "segment_ids", "indicator")})
    assert _rel(out[:, mt:].double(), out_ref[:, mt:].double()) < 1e-6      # the upstream forward returns float32


def test_peft_adapter_loads_into_peft():
    peft = pytest.importorskip("peft")
    tr = tiny_transformer(layers=2, dtype=F64)
    base = copy.deepcopy(tr)
    eng = E.FastDiT(copy.deepcopy(tr), rank=4, alpha=8.0, rslora=True, compute_dtype=F64, master_dtype=F64,
                    storage="bf16", attn="ref")
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        eng.lora.master.copy_(0.2 * torch.randn(eng.lora.master.shape, generator=g, dtype=F64))
    eng.lora.push()
    sd = eng.lora.state_peft()
    cfg = peft.LoraConfig(r=4, lora_alpha=8.0, target_modules=["qkv", "o", "w1", "w2", "w3"], use_rslora=True,
                          lora_dropout=0.0)
    pm = peft.get_peft_model(base, cfg)
    missing = peft.set_peft_model_state_dict(pm, sd)
    assert not [k for k in getattr(missing, "unexpected_keys", []) if "lora" in k]
    b = _batch(eng)
    mt = b["inp"]["mt"]
    B = b["z_t"].shape[0]
    x = torch.cat([torch.zeros(B, mt, 128, dtype=F64), b["z_t"]], 1)
    kw = dict(llm_features=b["llm"], x=x, t=b["t"], **{k: b["inp"][k] for k in ("position_ids", "segment_ids",
                                                                                 "indicator")})
    with torch.no_grad():
        want = pm(**kw)
    got = E.EngineModule(eng)(**kw)
    assert _rel(got[:, mt:].double(), want[:, mt:].double()) < 1e-6
    eng2 = E.FastDiT(copy.deepcopy(tr), rank=4, alpha=8.0, compute_dtype=F64, master_dtype=F64, storage="bf16",
                     attn="ref")
    eng2.lora.load_peft(sd)
    assert _rel(eng2.lora.master, eng.lora.master) < 1e-7    # float32 on disk


def test_fp8_codes_dequantise_to_the_reference_weights():
    tr = tiny_transformer(layers=1, dtype=F64)

    class F8:                                                 # what the ideogram4 package's Fp8Linear exposes
        def __init__(self, lin):
            self.weight, self.weight_scale = quantize_rows(lin.weight)
            self.bias = None

    blk = tr.layers[0]
    f8 = [F8(blk.feed_forward.w1), F8(blk.feed_forward.w3)]
    fz = E.frozen_from(f8, F64, "fp8")
    assert fz.codes is not None and fz.codes.dtype == torch.float8_e4m3fn
    want = torch.cat([(m.weight.float() * m.weight_scale.float()[:, None]).double() for m in f8])
    assert torch.equal(fz.weight(E.Scratch(), F64), want)
    fz2 = E.frozen_from(f8, F64, "bf16")                      # dequantised once at load: the same values
    assert torch.equal(fz2.w, want)


def test_flat_adamw_matches_per_tensor_adamw():
    _, eng = _models(layers=2)
    order = eng.lora.ref_order()
    params = {k: nn.Parameter(eng.lora.to_ref(eng.lora.master, n, kind).detach().clone()) for k, kind, n in order}
    a_keys = [k for k, kind, _ in order if kind == "A"]
    b_keys = [k for k, kind, _ in order if kind == "B"]
    ref = torch.optim.AdamW([{"params": [params[k] for k in a_keys], "lr": 1e-2},
                             {"params": [params[k] for k in b_keys], "lr": 1e-2 * 4.0}],
                            betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    opt, pa, pb = TR.make_optimizer(eng, 1e-2, 0.01, 4.0, fused=False)
    g = torch.Generator().manual_seed(3)
    for _ in range(3):
        for k, kind, n in order:
            gr = torch.randn(params[k].shape, generator=g, dtype=F64)
            params[k].grad = gr.clone()
            eng.lora.to_ref(eng.lora.grad, n, kind).copy_(gr)
        ref.step()
        opt.step()
    for k, kind, n in order:
        assert torch.allclose(eng.lora.to_ref(eng.lora.master, n, kind), params[k].detach(), rtol=1e-12, atol=1e-15)


def _run(tmp, *extra):
    args = ["--tiny", "--synthetic", "--device", "cpu", "--out", str(tmp), "--buckets", "64x48", "--batch_size", "2",
            "--grad_accum", "2", "--rank", "4", "--alpha", "4", "--max_steps", "6", "--warmup", "2", "--lr", "1e-3",
            "--log_every", "1", "--save_every", "3", "--synthetic_text_len", "6", *extra]
    assert TR.main(args) == 0
    return [json.loads(ln) for ln in (tmp / "log.jsonl").read_text().splitlines()]


def test_train_cli_end_to_end_and_exact_resume(tmp_path):
    from safetensors.torch import load_file
    full = _run(tmp_path / "full")
    steps = [r for r in full if "loss" in r and "event" not in r]
    assert [r["step"] for r in steps] == list(range(1, 7)) and all(math.isfinite(r["loss"]) for r in steps)
    cfg = json.loads((tmp_path / "full" / "step-0000006" / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == 4
    part = tmp_path / "part"
    _run(part, "--stop_after", "3")
    assert (part / "LATEST").read_text() == "step-0000003"
    _run(part, "--resume")
    a = load_file(str(tmp_path / "full" / "step-0000006" / "adapter_model.safetensors"))
    b = load_file(str(part / "step-0000006" / "adapter_model.safetensors"))
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(a[k], b[k], rtol=1e-5, atol=1e-7), k


def test_engine_on_the_ideogram4_packages_fp8_model():
    """With Ideogram's own package installed: a tiny transformer built by its FP8 loader (Fp8Linear modules) ->
    the stack's dequantisation of the non-target layers + the engine on FP8 codes == the same model fully
    dequantised, in float32."""
    pytest.importorskip("ideogram4")
    from ideogram4.modeling_ideogram4 import Ideogram4Transformer
    from ideogram4.pipeline_ideogram4 import _build_transformer
    from ideogram4.quantized_loading import quantize_weight_to_fp8
    from perf_kit.dit.ideogram4.stack import dequantize_except_targets, dequantize_fp8_linear
    tiny = tiny_transformer(layers=2, dtype=torch.float32)
    sd = {}
    for k, v in tiny.state_dict().items():
        if k.endswith(".weight") and v.dim() == 2 and "embed" not in k:
            q, sc = quantize_weight_to_fp8(v)
            sd[k], sd[k[:-len("weight")] + "weight_scale"] = q, sc
        else:
            sd[k] = v
    tr = _build_transformer(tiny.config, sd, torch.device("cpu"), torch.float32)
    assert isinstance(tr, Ideogram4Transformer)
    ref = copy.deepcopy(tr)
    for _, parent in list(ref.named_modules()):
        for cname, child in list(parent.named_children()):
            if hasattr(child, "weight_scale"):
                setattr(parent, cname, dequantize_fp8_linear(child, torch.float32))
    assert dequantize_except_targets(tr, torch.float32) > 0
    eng = E.FastDiT(tr, rank=4, alpha=4.0, compute_dtype=torch.float32, master_dtype=torch.float32, storage="fp8",
                    attn="ref")
    assert all(g.frozen.codes is not None for b in eng.blocks for g in b.groups.values())
    b = _batch(eng)
    mt = b["inp"]["mt"]
    B = b["z_t"].shape[0]
    kw = dict(llm_features=b["llm"].float(), x=torch.cat([torch.zeros(B, mt, 128), b["z_t"].float()], 1), t=b["t"],
              **{k: b["inp"][k] for k in ("position_ids", "segment_ids", "indicator")})
    from perf_kit.dit.ideogram4.stack import restore_rope_precision
    restore_rope_precision(ref)
    with torch.no_grad():
        want = ref(**kw)
    got = E.EngineModule(eng)(**kw)
    assert _rel(got[:, mt:], want[:, mt:]) < 1e-4
