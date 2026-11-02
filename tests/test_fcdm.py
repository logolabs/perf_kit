"""perf_kit.t2i.fcdm on CPU: the optimisations change the work, not the maths. The generator computes its position
features and rotary angles in float32 by design, so the model-level checks run in float32."""
from __future__ import annotations

import json
import math

import torch

from perf_kit.t2i.fcdm import model as M
from perf_kit.t2i.fcdm import train as TR
from perf_kit.t2i.fcdm.ema import EMA
from perf_kit.t2i.fcdm.flow import euler_sample, flow_loss
from perf_kit.t2i.fcdm.text import HashTextEncoder, compile_forward


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp(min=1e-300))


def _tiny(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    m = M.build("tiny")
    with torch.no_grad():                         # zero-init modulation would make the steering path inert
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return m.to(dtype)


def _inputs(hw=16, B=3, L=7, ctx_dim=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 4, hw, hw, generator=g)
    ctx = torch.randn(B, L, ctx_dim, generator=g)
    mask = torch.ones(B, L)
    mask[0, 5:] = 0
    t = torch.tensor([0.1, 0.5, 0.9])
    return x, t, ctx, mask, torch.randn(x.shape, generator=g)


def test_agate_preset_matches_the_release_size():
    with torch.device("meta"):
        m = M.build("agate")
    assert M.count_params(m) == 190_929_092          # Agate Preview 002's generator (its config.json)


def test_steer_upsample_once_is_the_same_function():
    m = _tiny()
    x, t, ctx, mask, noise = _inputs()
    outs, grads = [], []
    for once in (False, True):
        m.steer_upsample_once = once
        m.zero_grad()
        loss = flow_loss(m, x, t, ctx, mask, noise)
        loss.backward()
        outs.append(m(x, t, ctx, mask).detach())
        grads.append(torch.cat([p.grad.flatten() for p in m.parameters()]))
    assert _rel(outs[1], outs[0]) < 2e-6              # float32 rounding of a re-ordered sum
    assert _rel(grads[1], grads[0]) < 2e-5
    assert float(outs[0].abs().max()) > 0             # the path is not inert


def test_ema_foreach_is_bit_identical_and_skips_frozen():
    m = _tiny(dtype=torch.float32)
    frozen = list(m.thinker.parameters())
    for p in frozen:
        p.requires_grad_(False)
    loop, fe = EMA(m, 0.99, foreach=False), EMA(m, 0.99, foreach=True)
    for _ in range(3):
        with torch.no_grad():
            for p in m.parameters():
                if p.requires_grad:
                    p.add_(0.01 * torch.randn_like(p))
        loop.update(m)
        fe.update(m)
    for a, b in zip(loop.shadow.parameters(), fe.shadow.parameters()):
        assert torch.equal(a, b)
    n, diff, skipped = fe.check_foreach(m)
    assert diff == 0.0 and skipped == len(frozen) and n == len(list(m.parameters()))


def test_compile_forward_keeps_parameter_names():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.Linear(4, 2))
    keys = list(m.state_dict())
    calls = []

    def fake_compile(fn):                            # inductor needs a C++ toolchain; the contract is the wrapping
        def run(*a, **k):
            calls.append(1)
            return fn(*a, **k)
        return run
    x = torch.randn(3, 4)
    want = m(x)
    compile_forward(m, fake_compile)
    assert torch.equal(m(x), want) and calls
    assert list(m.state_dict()) == keys              # no _orig_mod. prefix
    m.load_state_dict(torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(), torch.nn.Linear(4, 2)).state_dict())


def test_release_round_trip(tmp_path):
    from safetensors.torch import save_file
    m = _tiny(dtype=torch.float32)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp_path / "generator.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"arch": "fcdm_t2", "model_kw": M.TINY}))
    m2, cfg, root = M.load_release(str(tmp_path))
    for (k, a), (k2, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert k == k2 and torch.equal(a, b)
    x, t, ctx, mask, _ = _inputs()
    z = euler_sample(m2.eval(), x.float(), ctx.float(), mask, ctx[:1, :1].float(), mask[:1, :1], steps=3, cfg=2.0)
    assert z.shape == x.shape and torch.isfinite(z).all()


def _shards(d, n=24, hw=16, seed=0):
    from safetensors.torch import save_file
    d.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    for s in range(2):
        save_file({"latents": torch.randn(n // 2, 4, hw, hw, generator=g).half()}, str(d / f"shard-{s:05d}.safetensors"))
        (d / f"shard-{s:05d}.jsonl").write_text("".join(json.dumps({"caption": f"a red cube number {s}-{i}"}) + "\n"
                                                        for i in range(n // 2)))
    return d


def _run(out, data, *extra):
    args = ["--preset", "tiny", "--device", "cpu", "--data", str(data), "--out", str(out), "--global_batch", "8",
            "--micro_batch", "4", "--lr", "1e-3", "--warmup", "2", "--max_steps", "6", "--log_every", "1",
            "--save_every", "3", "--ema_decay", "0.9", *extra]
    assert TR.main(args) == 0
    return [json.loads(ln) for ln in (out / "log.jsonl").read_text().splitlines()]


def test_train_cli_end_to_end_and_exact_resume(tmp_path):
    data = _shards(tmp_path / "data")
    full = _run(tmp_path / "full", data, "--steer_upsample_once")
    steps = [r for r in full if "loss" in r and "event" not in r]
    assert [r["step"] for r in steps] == list(range(1, 7)) and all(math.isfinite(r["loss"]) for r in steps)
    chk = next(r for r in full if r.get("event") == "ema_foreach_check")
    assert chk["max_abs_diff"] == 0.0
    model_ev = next(r for r in full if r.get("event") == "model")
    assert model_ev["train_gflop_per_img"] > 0 and model_ev["accum"] == 2
    assert (tmp_path / "full" / "export" / "generator.safetensors").exists()
    m2, _, _ = M.load_release(str(tmp_path / "full" / "export"))          # the export is a loadable release
    part = tmp_path / "part"
    _run(part, data, "--steer_upsample_once", "--stop_after", "3")
    assert (part / "LATEST").read_text() == "step-0000003.pt"
    _run(part, data, "--steer_upsample_once", "--resume")
    a = torch.load(tmp_path / "full" / "step-0000006.pt", weights_only=False)
    b = torch.load(part / "step-0000006.pt", weights_only=False)
    for k in a["model"]:
        assert torch.allclose(a["model"][k], b["model"][k], rtol=1e-5, atol=1e-6), k
        assert torch.allclose(a["ema"][k], b["ema"][k], rtol=1e-5, atol=1e-6), k


def test_hash_text_encoder_masks_and_pads():
    te = HashTextEncoder(dim=8, max_len=16)
    ids, mask = te.tokenize(["abc", ""], pad_to=8)
    assert ids.shape == (2, 8) and mask.sum(1).tolist() == [4, 1]
