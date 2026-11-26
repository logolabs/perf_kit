"""CPU tests of perf_kit.elastic (gloo, several local processes): fixed world vs the reference loop, join, leave,
crash, hang, resume from checkpoint, start from a pool version, per-GPU lr rule, hold. Each process = one 'job'."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
TINY = ["--ctx", "32", "--dim", "32", "--layers", "2", "--heads", "2", "--batch", "4", "--val_windows", "64",
        "--log_every", "1", "--chk_every", "5", "--lr", "3e-3", "--cpu"]
FAST = ["--hb_s", "0.5", "--dead_s", "4", "--gather_s", "15", "--found_wait_s", "2", "--regroup_min_s", "0",
        "--check_every_s", "0.5", "--timeout_s", "20", "--bf16_hook", "0"]


@pytest.fixture()
def data(tmp_path):
    rnd = random.Random(0)
    words = "the quick brown fox jumps over a lazy dog while the elastic group keeps changing".split()
    for name, n in (("train", 300), ("val", 30)):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps({"prompt": " ".join(rnd.choice(words) for _ in range(40)),
                                    "completion": " ".join(rnd.choice(words) for _ in range(40))}) + "\n")
    return ["--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl")]


def _env(job, extra=None):
    return {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "USE_LIBUV": "0",
            "CUDA_VISIBLE_DEVICES": "", "ELASTIC_JOB": job, "ELASTIC_IFACE": "lo", **(extra or {})}


def _start(mode, out, data, args, job, env_extra=None):
    cmd = [sys.executable, "-m", "perf_kit.elastic.example", "--mode", mode, "--out", str(out), *data, *TINY,
           *FAST, *map(str, args)]
    return subprocess.Popen(cmd, cwd=str(ROOT), env=_env(job, env_extra), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _result(p, timeout=600):
    o, e = p.communicate(timeout=timeout)
    lines = [x for x in o.splitlines() if x.startswith("RESULT ")]
    return (json.loads(lines[-1][7:]) if lines else None), o, e


def _curve(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        by[r["step"]] = r                                   # a redone step after a crash: the later row wins
    return by


def _events(run):
    p = Path(run) / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _ref(tmp_path, data, total, extra=()):
    out = tmp_path / "ref"
    r, o, e = _result(_start("ref", out, data, ["--total_samples", total, *extra], "ref"))
    assert r is not None, e[-3000:]
    return r, _curve(out / "metrics" / "ref.jsonl")


def _compare(cur_el, cur_ref, tol):
    common = sorted(set(cur_el) & set(cur_ref))
    assert len(common) >= 0.9 * len(cur_ref), (len(common), len(cur_ref))
    d = max(abs(cur_el[s]["loss"] - cur_ref[s]["loss"]) for s in common)
    assert d < tol, d
    return d


def _check_all(res):
    for r, o, e in res:
        assert r is not None, (o[-2000:], e[-4000:])


# ------------------------------------------------------------------ units
def test_micro_split_and_lr_rule():
    from perf_kit.elastic.core import ElasticTCP
    el = ElasticTCP.__new__(ElasticTCP)
    el.global_micro, el.accum, el.micro = 12, 1, 120
    el.micro_batch, el.ref_batch, el.lr_rule = 4, None, "sqrt"
    for W in (1, 2, 3, 4, 5, 7, 12):
        el.world = W
        got = []
        for r in range(W):
            el.rank = r
            got += list(el.micro_indices())
        assert got == list(range(120, 132)), (W, got)        # every micro-batch of the step exactly once
    assert el.lr_scale == 1.0                                # fixed global batch: no lr change with the world
    el.global_micro, el.accum, el.ref_batch, el.world = None, 2, 8, 4
    assert el.G == 8 and el.global_batch == 32 and abs(el.lr_scale - 2.0) < 1e-12   # sqrt(32 / 8)
    el.lr_rule = "linear"
    assert abs(el.lr_scale - 4.0) < 1e-12


def test_claim_has_one_winner(tmp_path):
    from perf_kit.elastic.membership import Membership
    ms = [Membership(tmp_path, f"m{i}", {"ip": "127.0.0.1", "job": f"j{i}", "joined": time.time()}) for i in range(5)]
    assert [m.claim(3) for m in ms].count(True) == 1
    assert ms[0].latest_epoch() == 3


def test_state_transfer_restores_optimizer_exactly(tmp_path):
    """broadcast_state between two gloo ranks: a fresh optimizer (no state) ends bit-identical to the source's."""
    script = tmp_path / "bc.py"
    script.write_text('''
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, sys.argv[3])
from perf_kit.elastic.state import broadcast_state, fingerprint
r = int(sys.argv[1]); dist.init_process_group("gloo", init_method="tcp://127.0.0.1:" + sys.argv[2], rank=r, world_size=2)
torch.manual_seed(r)
m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 2))
opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
if r == 0:
    for _ in range(3):
        m(torch.randn(4, 8)).pow(2).mean().backward(); opt.step(); opt.zero_grad()
got = broadcast_state({"model": m}, opt, {"step": 3 if r == 0 else 0}, {"cursor": 7} if r == 0 else {}, 0,
                      torch.device("cpu"))
st = opt.state_dict()["state"]
s = sum(float(v["exp_avg_sq"].double().sum()) for v in st.values())
print("OUT", fingerprint({"model": m}), got["counters"]["step"], got["extra"].get("cursor"), repr(s),
      float(next(iter(st.values()))["step"]))
dist.destroy_process_group()
''')
    port = str(29600 + os.getpid() % 500)
    ps = [subprocess.Popen([sys.executable, str(script), str(r), port, str(ROOT)], env=_env("x"),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for r in (0, 1)]
    outs = [p.communicate(timeout=120) for p in ps]
    lines = [[x for x in o.splitlines() if x.startswith("OUT")] for o, _ in outs]
    assert all(lines), [e[-2000:] for _, e in outs]
    a, b = lines[0][0].split(), lines[1][0].split()
    assert a == b and a[2] == "3" and a[3] == "7" and a[5] == "3.0", (a, b)


# ------------------------------------------------------------------ processes
def test_fixed_world_equals_reference(tmp_path, data):
    """3 workers that start together follow the 1-process reference (G = 12 micro-batches per step) to float noise;
    the parameter checksums of the 3 replicas agree at every check, the final weights are close."""
    total = 12 * 4 * 30
    run = tmp_path / "run"
    ps = [_start("elastic", run, data, ["--total_samples", total], f"j{i}") for i in range(3)]
    res = [_result(p) for p in ps]
    _check_all(res)
    rref, cref = _ref(tmp_path, data, total)
    cur = _curve(run / "metrics" / "elastic.jsonl")
    d = _compare(cur, cref, 2e-3)
    assert all(r["world"] == 3 for r in cur.values()), {r["world"] for r in cur.values()}
    assert sum(1 for r in cur.values() if r.get("chk_equal")) >= 5
    fin = [r for r, _, _ in res if r.get("val_bpb") is not None]
    assert len(fin) == 1 and abs(fin[0]["val_bpb"] - rref["val_bpb"]) < 5e-3, (fin, rref["val_bpb"], d)
    assert (run / "DONE").exists()


def test_join_leave_crash_equal_reference(tmp_path, data):
    """A starts alone; B joins; C joins; B announces leaving after 30 of its steps; C dies (os._exit) in its 45th
    step; D joins late.
    The world goes 1 -> 2 -> 3 -> 2 -> 1 -> 2 (or similar) and the loss curve still equals the reference: the fixed
    global batch makes the math independent of who is there, and a crashed step is redone on the same indices."""
    total = 12 * 4 * 120
    run = tmp_path / "run"
    sl = ["--total_samples", total, "--step_sleep", 0.15]
    ps = [_start("elastic", run, data, sl, "jA"),
          _start("elastic", run, data, [*sl, "--start_delay", 4, "--leave_at_step", 30], "jB"),
          _start("elastic", run, data, [*sl, "--start_delay", 6, "--die_at_step", 45], "jC"),
          _start("elastic", run, data, [*sl, "--start_delay", 14], "jD")]
    res = [_result(p) for p in ps]
    _check_all([res[0], res[1], res[3]])
    assert ps[2].returncode == 9, (res[2][1][-2000:], res[2][2][-3000:])
    assert res[1][0]["left"] and res[1][0]["steps_here"] >= 30
    _, cref = _ref(tmp_path, data, total)
    cur = _curve(run / "metrics" / "elastic.jsonl")
    d = _compare(cur, cref, 3e-3)
    ev = _events(run)
    kinds = [e["event"] for e in ev]
    assert "recovery" in kinds and "regroup" in kinds and kinds.count("epoch") >= 4, kinds
    worlds = [e["world"] for e in ev if e["event"] == "epoch"]
    assert max(worlds) >= 3, worlds
    rec = [e for e in ev if e["event"] == "recovery"][0]
    assert rec["lost"] and all(m.startswith("jC") for m in rec["lost"]), rec
    assert (run / "DONE").exists()
    print("max loss diff vs reference", d, "worlds", worlds)


def test_hung_rank_times_out_and_rejoins(tmp_path, data):
    """B sleeps 35 s inside a step (timeout 8 s): A times out, regroups alone (B is alive but not checked in, so A
    waits gather_s, then goes on) and finishes; curve = reference. B either wakes, fails and comes back as a
    newcomer (NCCL, measured on the cluster) or - gloo on Windows: its DDP all-reduce with the vanished peer never
    returns - is ended by the stall watchdog (exit 75, stall_s = 3 x 8 + 6 = 30 s < 35 s)."""
    total = 12 * 4 * 50
    run = tmp_path / "run"
    over = ["--timeout_s", 8, "--gather_s", 6]
    ps = [_start("elastic", run, data, ["--total_samples", total, *over], "jA"),
          _start("elastic", run, data, ["--total_samples", total, *over, "--hang_at_step", 10, "--hang_s", 35],
                 "jB")]
    res = [_result(p, timeout=300) for p in ps]
    _check_all([res[0]])
    _, cref = _ref(tmp_path, data, total)
    _compare(_curve(run / "metrics" / "elastic.jsonl"), cref, 3e-3)
    kinds = [e["event"] for e in _events(run)]
    assert "recovery" in kinds, kinds
    assert ps[1].returncode == 75 or (res[1][0] and (res[1][0]["finished"] or res[1][0]["left"])), res[1][2][-2000:]


def test_all_leave_then_resume_from_checkpoint(tmp_path, data):
    """Both workers announce leaving after 20 steps: rank 0 checkpoints; a new job later founds the run from
    that checkpoint and finishes it; the curve equals the reference over the whole run."""
    total = 12 * 4 * 45
    run = tmp_path / "run"
    ps = [_start("elastic", run, data, ["--total_samples", total, "--leave_at_step", 20], f"j{i}") for i in range(2)]
    res = [_result(p) for p in ps]
    _check_all(res)
    assert all(r["left"] for r, _, _ in res)
    assert list((run / "ckpt").glob("step_*.pt")) and not (run / "DONE").exists()
    r3 = _result(_start("elastic", run, data, ["--total_samples", total], "j3"))
    _check_all([r3])
    assert r3[0]["finished"]
    _, cref = _ref(tmp_path, data, total)
    _compare(_curve(run / "metrics" / "elastic.jsonl"), cref, 3e-3)
    found = [e for e in _events(run) if e["event"] == "found"]
    assert len(found) == 2 and "checkpoint" in found[1]["from"], found


def test_start_from_pool_version(tmp_path, data):
    """init_from='pool:DIR': the founder loads the pool's latest global weights (names cleaned) before epoch 0."""
    from perf_kit.pool.example_lm import GPT
    from perf_kit.pool.outer import OuterState
    from perf_kit.pool.store import Store
    from perf_kit.elastic.state import fingerprint
    torch.manual_seed(123)
    m = GPT(32, 32, 2, 2)
    pool = tmp_path / "pool"
    st = Store(pool)
    params = {k: p.detach().clone().float() for k, p in m.named_parameters()}
    os_ = OuterState(version=5, rounds=5.0, samples=999, params=params, buffers={}, m=None, buf=None, acc=0.0,
                     high_water={})
    assert st.commit(os_)
    run = tmp_path / "run"
    r = _result(_start("elastic", run, data, ["--total_samples", 48, "--init_from", f"pool:{pool}",
                                              "--lr", "0"], "j0"))
    _check_all([r])
    found = [e for e in _events(run) if e["event"] == "found"][0]
    assert "pool" in found["from"] and "version 5" in found["from"], found
    ck = sorted((run / "ckpt").glob("step_*.pt"))[-1]
    d = torch.load(ck, map_location="cpu", weights_only=False)
    m2 = GPT(32, 32, 2, 2)
    m2.load_state_dict(d["modules"]["model"])
    assert fingerprint({"model": m2}) == fingerprint({"model": m})      # lr 0: weights unchanged from the pool


def test_per_gpu_mode_scales_lr_with_world(tmp_path, data):
    """global_micro 0 (per-GPU batch fixed): 2 workers, accum 2 -> G = 4, global batch 16, ref_batch 4 ->
    lr_scale sqrt(4) = 2; samples counted globally."""
    run = tmp_path / "run"
    args = ["--total_samples", 16 * 20, "--global_micro", 0, "--accum", 2, "--ref_batch", 4]
    ps = [_start("elastic", run, data, args, f"j{i}") for i in range(2)]
    res = [_result(p) for p in ps]
    _check_all(res)
    cur = _curve(run / "metrics" / "elastic.jsonl")
    two = [r for r in cur.values() if r["world"] == 2]
    assert two and all(abs(r["lr_scale"] - 2.0) < 1e-9 and r["global_batch"] == 16 for r in two), two[:2]
    assert max(r["samples"] for r in cur.values()) == 16 * 20


def test_settings_mismatch_is_refused(tmp_path, data):
    run = tmp_path / "run"
    p1 = _start("elastic", run, data, ["--total_samples", 480], "j0")
    time.sleep(4)
    r2 = _result(_start("elastic", run, data, ["--total_samples", 480, "--global_micro", 6], "j1"))
    assert r2[0] is None and "differ from the run's config.json" in r2[2]
    _check_all([_result(p1)])


def test_hold_outlasts_the_collective_timeout(tmp_path, data):
    """Rank 0 works 25 s inside el.hold() (timeout 8 s, stall_s 30 s): the others wait on the store, nobody fails."""
    total = 12 * 4 * 30
    run = tmp_path / "run"
    over = ["--total_samples", total, "--timeout_s", 8, "--gather_s", 6, "--hold_at_step", 10, "--hold_s", 25]
    ps = [_start("elastic", run, data, over, f"j{i}") for i in range(2)]
    res = [_result(p, timeout=300) for p in ps]
    _check_all(res)
    kinds = [e["event"] for e in _events(run)]
    assert "recovery" not in kinds and "stall_exit" not in kinds and kinds.count("epoch") == 1, kinds
    assert all(r["finished"] for r, _, _ in res)


@pytest.mark.parametrize("late", [False, True])
def test_ema_updated_in_post_commit_survives_a_join(tmp_path, data, late):
    """An EMA (extra module) updated in step(post_commit=...) stays identical on all ranks across a join; updated
    after step() returns (the font generator's first version) the newcomer's EMA misses one step and the checksum
    check catches it (desync event + re-broadcast)."""
    total = 12 * 4 * 60
    run = tmp_path / "run"
    base = ["--total_samples", total, "--ema", 0.9, "--chk_every", 1, "--step_sleep", 0.1, *(["--ema_late"] if late
                                                                                            else [])]
    ps = [_start("elastic", run, data, base, "jA"), _start("elastic", run, data, [*base, "--start_delay", 6], "jB")]
    res = [_result(p) for p in ps]
    _check_all(res)
    ev = _events(run)
    assert max(e["world"] for e in ev if e["event"] == "epoch") == 2
    n_desync = sum(e["event"] == "desync" for e in ev)
    assert (n_desync >= 1) if late else (n_desync == 0), [e["event"] for e in ev]
