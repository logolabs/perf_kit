"""CPU unit tests: layout (reference step sets, LPT balance, micro split), packing, block-FP8 dequant, GroupLinear +
LoraStore (fused LoRA forward/backward = PEFT math in float64; flat store push / PEFT round trip), the checkpoint
policy, the engine step (equal update whatever the layout)."""
import random

import numpy as np
import pytest
import torch

from perf_kit import layout as Lo
from perf_kit.engine import CkptPolicy
from perf_kit.kernels.dequant import dequant_into, dequant_torch
from perf_kit.lora import GroupLinear, LoraStore, Member, Scratch
from perf_kit.packing import IGNORE, pack


def test_reference_steps_match_classic_rank_interleave():
    rng = random.Random(0)
    lengths = [rng.randint(20, 900) for _ in range(500)]
    micro, world, accum, seed = 2, 4, 4, 0
    src = Lo.reference_steps(lengths, micro, world, accum, seed)
    mbs = Lo.micro_batches(lengths, micro, seed)
    mine = [mbs[r::world] for r in range(world)]
    for s in (0, 1, 7, src.steps_per_epoch - 1, src.steps_per_epoch + 3):     # incl. wrap-around (2nd epoch)
        want = []
        for r in range(world):
            for j in range(accum):
                want += mine[r][(s * accum + j) % len(mine[r])]
        assert src.step(s) == want


def test_balance_is_near_perfect_and_keeps_the_step_set():
    rng = random.Random(1)
    lengths = [rng.randint(300, 1600) for _ in range(64)]
    rows = list(range(32))
    per = Lo.balance(rows, lengths, 4)
    assert sorted(sum(per, [])) == rows
    assert Lo.imbalance(per, lengths) < 1.03
    lay = Lo.plan_step(rows, lengths, [5] * 64, 2, 4, max_tokens=2000)
    assert sorted(sum(lay.micro, [])) == sorted(per[2]) and all(sum(lengths[i] for i in b) <= 2000 or len(b) == 1
                                                                for b in lay.micro)
    assert lay.label_total == 5 * 32


def test_split_micro_first_fit_decreasing():
    lengths = [900, 800, 700, 300, 200, 100, 5000]
    bins = Lo.split_micro(list(range(7)), lengths, 1000)
    assert [6] in bins                              # too long: alone
    assert all(sum(lengths[i] for i in b) <= 1000 for b in bins if b != [6])
    assert sorted(sum(bins, [])) == list(range(7))


def test_pack_targets_never_cross_rows():
    rows = [([1, 2, 3, 4], [IGNORE, IGNORE, 3, 4], [0, 0, 2, 2]), ([5, 6, 7], [IGNORE, 6, 7], [0, 1, 2])]
    p = pack(rows)
    assert p.ids.tolist() == [1, 2, 3, 4, 5, 6, 7] and p.pos.tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert p.tgt.tolist() == [IGNORE, 3, 4, IGNORE, 6, 7, IGNORE]
    assert p.seg.tolist()[:3] == [0, 2, 2] and p.seg.tolist()[4:6] == [1, 2]
    assert p.cu.tolist() == [0, 4, 7] and p.keep.tolist() == [1, 2, 4, 5] and p.max_len == 4


def test_dequant_torch_path_equals_reference_and_writes_slices():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(300, 260, generator=g) * 0.05).to(torch.float8_e4m3fn)
    s = (torch.rand(3, 3, generator=g) + 0.5).to(torch.bfloat16)
    want = (w.float() * s.float().repeat_interleave(128, 0)[:300].repeat_interleave(128, 1)[:, :260]).to(torch.bfloat16)
    assert torch.equal(dequant_torch(w, s), want)
    buf = torch.zeros(500, 260, dtype=torch.bfloat16)
    dequant_into(buf[100:400], w, s)
    assert torch.equal(buf[100:400], want) and buf[:100].abs().sum() == 0 and buf[400:].abs().sum() == 0


def _group(dtype=torch.float64, storage="bf16"):
    g = torch.Generator().manual_seed(3)
    ms = [Member("a.q", 12, lora=True), Member("a.k", 6, lora=False), Member("a.v", 9, lora=True)]
    for m in ms:
        m.w = torch.randn(m.out, 10, generator=g, dtype=dtype)
    grp = GroupLinear(ms, 10, 4, 2.0, storage, Scratch(), "cpu", dtype)
    st = LoraStore([grp], 4, 2.0, "cpu", compute_dtype=dtype, master_dtype=dtype)
    with torch.no_grad():
        for m in st.slots.values():
            m.a_master.copy_(torch.randn(m.a_master.shape, generator=g, dtype=dtype))
            m.b_master.copy_(torch.randn(m.b_master.shape, generator=g, dtype=dtype))
    st.push()
    return grp, st, ms


def test_group_linear_forward_backward_equals_peft_math():
    grp, st, ms = _group()
    W = torch.cat([m.w if m.w is not None else grp.w[m.off:m.off + m.out] for m in ms]) if grp.w is None else grp.w
    x = torch.randn(7, 10, dtype=torch.float64, requires_grad=True)
    anchor = torch.zeros((), dtype=torch.float64, requires_grad=True)
    y = grp(x, anchor)
    dy = torch.randn_like(y)
    y.backward(dy)
    # reference: per-member Linear + PEFT LoRA
    x2 = x.detach().clone().requires_grad_(True)
    As = {n: m.a_master.detach().clone().requires_grad_(True) for n, m in st.slots.items()}
    Bs = {n: m.b_master.detach().clone().requires_grad_(True) for n, m in st.slots.items()}
    outs = []
    for m in ms:
        o = x2 @ W[m.off:m.off + m.out].t()
        if m.lora:
            o = o + (x2 @ As[m.name].t() @ Bs[m.name].t()) * 2.0
        outs.append(o)
    ref = torch.cat(outs, 1)
    ref.backward(dy)
    assert torch.allclose(y, ref, atol=1e-12)
    assert torch.allclose(x.grad, x2.grad, atol=1e-12)
    for n, m in st.slots.items():
        assert torch.allclose(m.a_grad, As[n].grad, atol=1e-12) and torch.allclose(m.b_grad, Bs[n].grad, atol=1e-12)


def test_lora_store_peft_round_trip_and_push():
    grp, st, ms = _group()
    sd = st.state_peft()
    assert set(sd) == {f"base_model.model.{n}.lora_{ab}.weight" for n in ("a.q", "a.v") for ab in "AB"}
    grp2, st2, _ = _group()
    st2.master.zero_()
    st2.load_peft(sd)
    assert torch.equal(st2.master, st.master)
    assert torch.equal(st2.compute[st2.n_a:], st2.master[st2.n_a:] * 2.0)


def test_ckpt_policy_budget():
    p = CkptPolicy(64, budget_gb=10.0, act_mb=0.25, in_mb=0.01)
    assert sum(p.plan(500)) == 0                     # 500 tok x 64 layers x 0.25 MB = 8 GB fits in 10 GB
    plan = p.plan(1000)                              # 16 GB: (16000 - 10240) / (250 - 10) MB -> 24 layers recomputed
    on = [i for i, f in enumerate(plan) if f]
    assert len(on) == 24 and max(b - a for a, b in zip(on, on[1:])) <= 3
    assert sum(CkptPolicy(64, 1.0, mode="none").plan(10 ** 6)) == 0 and all(CkptPolicy(4, 99, mode="all").plan(1))


def test_gate_verdict(tmp_path):
    import json
    from perf_kit import gate
    mine = [{"step": s, "step_s": 2.0, "mfu": 0.33} for s in range(10, 310, 10)]
    mine += [{"event": "val", "step": 300, "val_answer_loss": 0.6870}, {"event": "profile", "gpu_idle": 0.05}]
    ref = [{"step": s, "step_s": 5.0, "mfu": 0.13} for s in range(10, 310, 10)]
    ref += [{"event": "val", "step": 300, "val_answer_loss": 0.6866}]
    p, q = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in mine))
    q.write_text("\n".join(json.dumps(r) for r in ref))
    v = gate.verdict(gate.load(p), 0.30, gate.load(q), 0.005, 0.1)
    assert v["pass"] and v["speedup_vs_ref"] == 2.5 and v["checks"] == {"mfu": True, "val": True, "idle": True}
    assert not gate.verdict(gate.load(p), 0.40)["pass"]
    assert gate.main([str(p), "--mfu_min", "0.3", "--ref", str(q)]) == 0


def test_fp8_group_runs_equal_bf16_group():
    g = torch.Generator().manual_seed(5)

    def members():
        out = []
        for name, n, fp8 in (("x.qkv", 256, True), ("x.z", 128, True), ("x.b", 3, False), ("x.a", 3, False)):
            m = Member(name, n, lora=name in ("x.qkv", "x.z"))
            w = torch.randn(n, 200, generator=g) * 0.05
            if fp8:
                m.codes = w.to(torch.float8_e4m3fn)
                m.scale_inv = torch.rand((n + 127) // 128, 2, generator=g) + 0.5
            else:
                m.w = w.to(torch.bfloat16)
            out.append(m)
        return out
    ms = members()
    want = torch.cat([dequant_torch(m.codes, m.scale_inv) if m.codes is not None else m.w for m in ms])
    grp = GroupLinear(ms, 200, 4, 2.0, "fp8", Scratch(), "cpu")
    assert [r["kind"] for r in grp.runs] == ["fp8", "bf16"] and grp.runs[0]["rows"] == 384
    assert torch.equal(grp.weight(), want)


def test_fp8_forward_fallback_equals_bf16_path():
    g = torch.Generator().manual_seed(7)

    def grp(fp8_fwd):
        ms = []
        for name, n in (("y.gate_proj", 256), ("y.up_proj", 256)):
            m = Member(name, n, lora=True)
            m.codes = (torch.randn(n, 256, generator=torch.Generator().manual_seed(n)) * 0.05).to(torch.float8_e4m3fn)
            m.scale_inv = torch.full((n // 128, 2), 0.5)
            ms.append(m)
        G = GroupLinear(ms, 256, 4, 2.0, "fp8", Scratch(), "cpu", torch.float32)
        st = LoraStore([G], 4, 2.0, "cpu", compute_dtype=torch.float32)
        st.init_new(0)
        G.fp8_fwd = fp8_fwd
        return G
    x = torch.randn(9, 256, generator=g)
    a = torch.zeros((), requires_grad=True)
    assert torch.allclose(grp(True)(x, a), grp(False)(x, a), atol=1e-5)    # CPU fallback = dequant + matmul


def test_ckpt_policy_calibrated_from_the_measured_9b_value():
    """The 9B case that motivated the calibration: 32 layers, ~57 GB budget. With the old fixed 0.25 MiB/token/layer
    the plan recomputes at 8,192 tokens although the measured 0.17 fits ~10.7k tokens without any recompute."""
    old = CkptPolicy(32, budget_gb=57.3, act_mb=0.25, in_mb=0.01)
    assert sum(old.plan(8192)) > 0 and old.max_tokens_without_recompute() < 7400
    new = CkptPolicy(32, budget_gb=57.3).calibrate(0.17, 0.016, margin=1.05)
    assert new.calibrated and abs(new.act_mb - 0.1785) < 1e-9
    assert sum(new.plan(8192)) == 0 and sum(new.plan(10000)) == 0 and sum(new.plan(12288)) > 0
    assert 10000 < new.max_tokens_without_recompute() < 10500
    with pytest.raises(ValueError):
        CkptPolicy(4, 1.0).calibrate(0.01, 0.02)          # a checkpointed layer cannot store more than a full one


def test_measure_act_mb_cpu_is_per_token_and_per_layer():
    import transformers
    from perf_kit.engine import measure_act_mb
    from perf_kit.models.dense import FastDense
    from perf_kit.weights import DictSource
    cfg = transformers.Qwen3Config(vocab_size=101, hidden_size=48, intermediate_size=80, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=12,
                                   tie_word_embeddings=False, max_position_embeddings=1024)
    torch.manual_seed(0)
    sd = {k: v.detach() for k, v in transformers.Qwen3ForCausalLM(cfg).state_dict().items()}
    fm = FastDense(cfg, DictSource(sd), prefix="model.", rank=4, alpha=8, storage="bf16", device="cpu",
                   dtype=torch.float32, impl="ref")
    fm.lora.init_new(0)
    g = torch.Generator().manual_seed(0)

    def rows(n, L):
        out = []
        for _ in range(n):
            ids = torch.randint(0, 101, (L,), generator=g).tolist()
            out.append((ids, [IGNORE] * 3 + ids[3:], [2] * L))
        return out
    a = measure_act_mb(fm, pack(rows(2, 40)))
    b = measure_act_mb(fm, pack(rows(4, 40)))
    assert a["method"] == "saved_tensors" and a["tokens"] == 80 and b["tokens"] == 160
    assert a["act_mb"] > a["in_mb"] > 0
    assert abs(a["act_mb"] - b["act_mb"]) / b["act_mb"] < 0.05        # linear in tokens
    assert a["in_mb"] >= 2 * 48 * 4 / 2**20                            # at least the saved residual stream


def test_sequential_steps_take_consecutive_rows_and_wrap():
    s = Lo.SequentialSteps(n_rows=70, global_batch=32)
    assert s.steps_per_epoch == 2
    assert s.step(0) == list(range(32)) and s.step(1) == list(range(32, 64)) and s.step(2) == list(range(32))
    with pytest.raises(ValueError):
        Lo.SequentialSteps(10, 32).step(0)
