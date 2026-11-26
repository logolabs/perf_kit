"""perf_kit.pool: outer update math in float64 against references, the run folder's crash rules, and fake-worker
simulations in separate processes (join late, die mid-round, die holding the lock, pool cap)."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from perf_kit.pool import outer as O
from perf_kit.pool.store import Store

ROOT = Path(__file__).resolve().parent.parent
F64 = torch.float64


def _params(seed, shapes=((5, 3), (7,), (2, 2, 2))):
    g = torch.Generator().manual_seed(seed)
    return {f"p{i}": torch.randn(*s, generator=g, dtype=F64) for i, s in enumerate(shapes)}


def _state(seed=0):
    return O.OuterState(params=_params(seed))


def _contrib(worker, seq, delta, samples=100, base_rounds=0.0):
    return O.Contribution(worker=worker, seq=seq, samples=samples, base_rounds=base_rounds, base_version=0,
                          delta=delta)


# ------------------------------------------------------------------ outer math vs references
@pytest.mark.parametrize("mode", ["dn", "buffered"])
def test_lockstep_equals_diloco_algorithm1(mode):
    """k equal workers contributing once per round from the same base == DiLoCo Algorithm 1: mean pseudo-gradient
    as the gradient of torch.optim.SGD(lr=0.7, momentum=0.9, nesterov=True)."""
    k, T = 4, 6
    st = _state()
    ref = {n: p.clone().requires_grad_(True) for n, p in st.params.items()}
    opt = torch.optim.SGD(list(ref.values()), lr=0.7, momentum=0.9, nesterov=True)
    cfg = O.OuterConfig(lr=0.7, momentum=0.9, mode=mode)
    for t in range(T):
        deltas = [_params(100 * t + i) for i in range(k)]
        O.apply(st, [_contrib(f"w{i}", t, deltas[i], base_rounds=st.rounds) for i in range(k)], 4 * 100, cfg)
        for n, p in ref.items():
            p.grad = sum(d[n] for d in deltas) / k
        opt.step()
        for n in ref:
            torch.testing.assert_close(st.params[n], ref[n].detach(), rtol=0, atol=1e-13)
    assert st.outer_steps == T and st.acc == 0.0 and st.samples == T * k * 100


def test_export_is_the_heavy_ball_iterate():
    """export_params (stored weights minus the Nesterov look-ahead) == torch SGD(momentum, nesterov=False) run on
    the same mean pseudo-gradients, at every round."""
    k, T = 3, 7
    st = _state(5)
    ref = {n: p.clone().requires_grad_(True) for n, p in st.params.items()}
    opt = torch.optim.SGD(list(ref.values()), lr=0.7, momentum=0.9, nesterov=False)
    cfg = O.OuterConfig(lr=0.7, momentum=0.9)
    for t in range(T):
        deltas = [_params(900 + 10 * t + i) for i in range(k)]
        O.apply(st, [_contrib(f"w{i}", t, deltas[i], base_rounds=st.rounds) for i in range(k)], k * 100, cfg)
        for n, p in ref.items():
            p.grad = sum(d[n] for d in deltas) / k
        opt.step()
        x = O.export_params(st, cfg)
        for n in ref:
            torch.testing.assert_close(x[n], ref[n].detach(), rtol=0, atol=1e-13)


def _delayed_nesterov_reference(theta, gs, N, lr, beta, c=0.0):
    """Liu et al. 2024 (arXiv 2401.09135) Algorithm 3, written out literally."""
    theta = {k: v.clone() for k, v in theta.items()}
    m = {k: torch.zeros_like(v) for k, v in theta.items()}
    buf = {k: torch.zeros_like(v) for k, v in theta.items()}
    for t, g in enumerate(gs):
        for k in theta:
            buf[k] += g[k]
        if (t + 1) % N == 0:
            for k in theta:
                m[k] = beta * m[k] + buf[k] / N
                theta[k] = theta[k] - lr * ((1 - c * N + c) * beta * m[k] + g[k] / N)
                buf[k] = torch.zeros_like(buf[k])
        else:
            for k in theta:
                theta[k] = theta[k] - lr * (c * beta * m[k] + g[k] / N)
    return theta


def test_async_equals_delayed_nesterov_algorithm3():
    N, n_arrivals = 3, 17
    st = _state(1)
    theta0 = {k: v.clone() for k, v in st.params.items()}
    gs = [_params(500 + t) for t in range(n_arrivals)]
    cfg = O.OuterConfig(lr=0.7, momentum=0.9, mode="dn")
    for t, g in enumerate(gs):        # one arrival at a time, bases one round behind (normal async staleness)
        O.apply(st, [_contrib(f"w{t % N}", t, g, base_rounds=max(0.0, st.rounds - 1))], N * 100, cfg)
    ref = _delayed_nesterov_reference(theta0, gs, N, 0.7, 0.9)
    for k in ref:
        torch.testing.assert_close(st.params[k], ref[k], rtol=0, atol=1e-13)


def test_sample_weighted_mean():
    """momentum 0, lr 1: one round == the FedAvg sample-weighted mean of the local weights."""
    st = _state(2)
    th = {k: v.clone() for k, v in st.params.items()}
    locals_ = [_params(10), _params(11)]
    samples = [100, 300]
    cs = [_contrib(f"w{i}", 0, {k: th[k] - loc[k] for k in th}, samples=s) for i, (loc, s) in
          enumerate(zip(locals_, samples))]
    for mode in ("dn", "buffered"):
        s2 = O.OuterState(params={k: v.clone() for k, v in th.items()})
        O.apply(s2, cs, 400, O.OuterConfig(lr=1.0, momentum=0.0, mode=mode))
        for k in th:
            torch.testing.assert_close(s2.params[k], (100 * locals_[0][k] + 300 * locals_[1][k]) / 400,
                                       rtol=0, atol=1e-13)


def test_staleness_drop_and_poly():
    st = _state(3)
    st.rounds = 10.0
    before = {k: v.clone() for k, v in st.params.items()}
    log = O.apply(st, [_contrib("old", 0, _params(1), base_rounds=5.0)], 100, O.OuterConfig(max_staleness=4))
    assert log[0].status == "dropped" and st.samples == 0
    for k in before:
        assert torch.equal(st.params[k], before[k])
    log = O.apply(st, [_contrib("ok", 0, _params(1), base_rounds=7.0)], 400,
                  O.OuterConfig(staleness_weight="poly", lr=1.0, momentum=0.0))
    assert log[0].status == "applied" and abs(log[0].weight - 0.25 * (1 + 3.0) ** -0.5) < 1e-15


def test_duplicate_is_ignored():
    st = _state(4)
    c = _contrib("w", 3, _params(9))
    O.apply(st, [c], 1000, O.OuterConfig())
    snap = {k: v.clone() for k, v in st.params.items()}
    log = O.apply(st, [c, _contrib("w", 2, _params(8))], 1000, O.OuterConfig())
    assert [a.status for a in log] == ["duplicate", "duplicate"]
    for k in snap:
        assert torch.equal(st.params[k], snap[k])


# ------------------------------------------------------------------ the Pool API on one process
def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()


def _batch(i):
    g = torch.Generator().manual_seed(i)
    x = torch.randn(4, 6, generator=g, dtype=F64)
    return x, x.sum(1, keepdim=True).sin()


def test_single_worker_lr1_mom0_equals_plain_training(tmp_path):
    """One worker, outer lr 1, momentum 0: the pool is a no-op and training equals the plain loop step for step."""
    from perf_kit.pool import Pool
    steps = 23
    ref = _tiny_model()
    ropt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    for i in range(steps):
        x, y = _batch(i)
        ropt.zero_grad()
        torch.nn.functional.mse_loss(ref(x), y).backward()
        ropt.step()
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    pool = Pool(tmp_path / "run", m, inner_steps=5, total_samples=10 ** 9, outer_lr=1.0, outer_momentum=0.0,
                state_dtype=F64, worker_id="solo", log=lambda *a: None)
    for i in range(steps):
        x, y = _batch(i)
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        opt.step()
        pool.sync(m, 4)
    for a, b in zip(ref.parameters(), m.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-12)
    base = {k: v.clone() for k, v in pool._global.params.items()}
    pool.close(m)
    g = Store(tmp_path / "run").load()
    assert g.samples == steps * 4                       # the partial round (3 of 5 steps) was published by close()
    w = (3 * 4) / (5 * 4)                               # ... and counts its samples over a full round's
    for (n, p) in m.named_parameters():
        torch.testing.assert_close(g.params[n], base[n] - w * (base[n] - p.detach()), rtol=0, atol=1e-12)


def test_crash_between_commit_and_inbox_delete(tmp_path):
    """A version is committed but the applier dies before deleting the inbox: the next applier must not re-apply."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    pool = Pool(tmp_path / "run", m, inner_steps=1, total_samples=10 ** 9, state_dtype=F64, worker_id="a",
                log=lambda *a: None)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)
    pool._publish(m)
    pool._seq += 1
    st = pool.store
    files = st.pending()
    state = st.load()
    O.apply(state, [st.read_contribution(f) for f in files], pool.round_samples(), pool.ocfg)
    state.version += 1
    assert st.commit(state)                             # ... and "dies" here, inbox file left behind
    assert not st.commit(state)                         # version numbers are claimed exclusively
    v = st.latest()
    pool._apply(wait=5)
    assert st.load().version == v + 1 and st.load().samples == state.samples   # duplicate, nothing re-applied
    assert not st.pending()
    ev = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in ev if e["event"] in ("applied", "duplicate")] == ["duplicate"]
    pool.close()


def test_stale_lock_is_taken_over(tmp_path):
    st = Store(tmp_path / "run", lock_timeout=0.5)
    (tmp_path / "run" / "lock").write_text('{"holder": "dead"}')
    with pytest.raises(Exception):
        with st.lock(wait=0.2):
            pass
    time.sleep(0.6)
    with st.lock(wait=2):
        assert json.loads((tmp_path / "run" / "lock").read_text())["holder"] != "dead"
    assert not (tmp_path / "run" / "lock").exists()


# ------------------------------------------------------------------ multi-process simulations
def _spawn(run, worker, *extra):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"}
    return subprocess.Popen([sys.executable, "-m", "perf_kit.pool.sim", "--run", str(run), "--worker", worker,
                             *map(str, extra)], cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)


def _events(run):
    return [json.loads(x) for x in (Path(run) / "events.jsonl").read_text().splitlines()]


def _finish(procs, timeout=240):
    outs = {}
    for name, p in procs.items():
        o, e = p.communicate(timeout=timeout)
        outs[name] = (p.returncode, o, e)
    return outs


def test_sim_join_leave_die(tmp_path):
    """Three workers (1, 1 and 2 'GPUs'): w0 from the start, w1 joins late, w2 dies mid-round after 2 rounds.
    The run must finish, apply each contribution at most once, never count the dead worker's lost round, and
    converge on the regression problem."""
    run = tmp_path / "run"
    total = 12000
    procs = {"w0": _spawn(run, "w0", "--total_samples", total, "--slow", 0.002),
             "w2": _spawn(run, "w2", "--gpus", 2, "--total_samples", total, "--die_after_rounds", 2,
                          "--start_delay", 1.0, "--slow", 0.002),
             "w1": _spawn(run, "w1", "--total_samples", total, "--start_delay", 3.0, "--slow", 0.002)}
    outs = _finish(procs)
    assert outs["w2"][0] == 17, outs["w2"]
    for w in ("w0", "w1"):
        assert outs[w][0] == 0, outs[w][2][-2000:]
        r = json.loads(outs[w][1].strip().splitlines()[-1])
        assert Path(r["perf_kit"]).resolve() == (ROOT / "perf_kit").resolve()     # the worktree's code
        assert r["excess"] < 1e-3, r
    ev = _events(run)
    keys = [(e["worker"], e["seq"]) for e in ev if e["event"] == "applied"]
    assert len(keys) == len(set(keys))
    assert {w for w, _ in keys} == {"w0", "w1", "w2"}
    assert max(s for w, s in keys if w == "w2") == 1          # rounds 0..1 arrived; the 3rd never did
    done = json.loads((run / "DONE").read_text())
    assert done["samples"] >= total
    st = Store(run).load()
    assert st.samples == sum(e["samples"] for e in ev if e["event"] == "applied")
    joined = [e["worker"] for e in ev if e["event"] == "joined"]
    assert joined[0] == "w0" and set(joined) == {"w0", "w1", "w2"}


def test_sim_dies_holding_lock(tmp_path):
    """w0 dies while holding the run lock; w1 keeps going after the stale-lock takeover and finishes the run."""
    run = tmp_path / "run"
    total = 3000
    procs = {"w0": _spawn(run, "w0", "--total_samples", total, "--die_after_rounds", 2, "--die_in", "lock",
                          "--slow", 0.002),
             "w1": _spawn(run, "w1", "--total_samples", total, "--start_delay", 0.5, "--slow", 0.004)}
    outs = _finish(procs)
    assert outs["w0"][0] == 18
    assert outs["w1"][0] == 0, outs["w1"][2][-2000:]
    ev = _events(run)
    assert any(x["event"] == "lock_takeover" for x in ev)
    assert json.loads((run / "DONE").read_text())["samples"] >= total
    assert json.loads(outs["w1"][1].strip().splitlines()[-1])["excess"] < 1e-3


def test_sim_pool_cap(tmp_path):
    """max_gpus=2 with two live 1-GPU workers: a third is refused (exits without joining); after they finish and
    leave, the run is DONE and a newcomer is not admitted either."""
    run = tmp_path / "run"
    total = 10000                                     # the run lasts well past w3's attempt at ~3 s
    procs = {w: _spawn(run, w, "--total_samples", total, "--max_gpus", 2, "--slow", 0.01) for w in ("w0", "w1")}
    time.sleep(3.0)
    o, _ = _spawn(run, "w3", "--total_samples", total, "--max_gpus", 2).communicate(timeout=120)
    r3 = json.loads(o.strip().splitlines()[-1])
    outs = _finish(procs)
    assert all(outs[w][0] == 0 for w in outs), {w: outs[w][2][-1500:] for w in outs}
    assert r3["full"] and r3["rounds"] == 0
    ev = _events(run)
    assert [x["worker"] for x in ev if x["event"] == "refused"] == ["w3"]
    assert (run / "DONE").exists()


# ------------------------------------------------------------------ launcher (fake sbatch / squeue)
def test_launcher_cap_pending_budget(tmp_path):
    from perf_kit.pool import launch
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import json, sys, pathlib\n"
        "db = pathlib.Path(sys.argv[1]); d = json.loads(db.read_text()) if db.exists() else {'n': 100, 'st': {}}\n"
        "if sys.argv[2] == 'sbatch':\n"
        "    d['n'] += 1; d['st'][str(d['n'])] = 'PENDING'; db.write_text(json.dumps(d)); print(d['n'])\n"
        "else:\n"
        "    ids = sys.argv[sys.argv.index('-j') + 1].split(',')\n"
        "    for i in ids:\n"
        "        if d['st'].get(i) in ('PENDING', 'RUNNING'): print(i, d['st'][i])\n")
    db = tmp_path / "db.json"
    q = f'"{sys.executable}" "{fake}" "{db}"'
    run = tmp_path / "run"
    base = ["--run", str(run), "--sbatch", "job.sbatch", "--gpus", "2", "--time", "00:30:00", "--max_gpus", "6",
            "--max_pending", "2", "--max_jobs", "10", "--budget_gpu_hours", "4", "--sbatch_bin", q + " sbatch",
            "--squeue", q + " squeue"]
    def ns(extra=()):
        return launch.argparse.Namespace(**{**vars(launch.parser().parse_args(base)), **dict(extra)})
    r = launch.topup(ns())
    assert r["submitted"] == ["101", "102"] and r["stop"].startswith("2 pending")
    d = json.loads(db.read_text()); d["st"] = {"101": "RUNNING", "102": "RUNNING"}; db.write_text(json.dumps(d))
    r = launch.topup(ns())
    assert r["submitted"] == ["103"] and r["stop"].startswith("cap")          # 3 jobs x 2 GPUs = the cap of 6
    d = json.loads(db.read_text()); d["st"] = {}; db.write_text(json.dumps(d))  # all finished
    r = launch.topup(ns())
    assert r["submitted"] == ["104"] and r["stop"].startswith("budget")        # 3 GPU-h booked + 1 = 4 fits, a 5th not
    (run / "DONE").write_text("{}")
    assert launch.topup(ns(dict(budget_gpu_hours=100)))["stop"] == "run is DONE"
    assert launch._hours("1-00:30:00") == 24.5 and launch._hours("45") == 0.75



def test_ddp_inside_a_worker(tmp_path):
    """A 2-rank worker (gloo DDP inside one 'job') and a 1-rank worker on one run of the example LM: only rank 0
    touches files, every rank keeps the same step counters and learning rates (inner_steps 60 > done_check_every 50,
    so the mid-round refresh is broadcast too), all finish, the run is DONE."""
    import random
    rnd = random.Random(0)
    words = "the quick brown fox jumps over a lazy dog while the pool keeps growing".split()
    for name, n in (("train", 300), ("val", 30)):
        with open(tmp_path / f"{name}.jsonl", "w") as f:
            for _ in range(n):
                f.write(json.dumps({"prompt": " ".join(rnd.choice(words) for _ in range(40)),
                                    "completion": " ".join(rnd.choice(words) for _ in range(40))}) + "\n")
    run = tmp_path / "run"
    args = ["--cpu", "--mode", "pool", "--out", str(run), "--train", str(tmp_path / "train.jsonl"), "--val",
            str(tmp_path / "val.jsonl"), "--total_samples", "16000", "--batch", "8", "--ctx", "32", "--dim", "32",
            "--layers", "2", "--heads", "2", "--val_windows", "64", "--log_every", "10", "--inner_steps", "60"]
    base = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(29500 + os.getpid() % 1000)}
    script = str(ROOT / "perf_kit" / "pool" / "example_lm.py")
    ps = [subprocess.Popen([sys.executable, script, *args], cwd=str(ROOT), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True,
                           env={**base, "RANK": str(r), "WORLD_SIZE": "2", "LOCAL_RANK": "0"}) for r in (0, 1)]
    time.sleep(1.0)
    ps.append(subprocess.Popen([sys.executable, script, *args], cwd=str(ROOT), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env={**base, "RANK": "0", "WORLD_SIZE": "1"}))
    outs = [p.communicate(timeout=300) for p in ps]
    assert all(p.returncode == 0 for p in ps), [o[1][-1500:] for o in outs]
    ev = _events(run)
    joined = {e["worker"]: e["gpus"] for e in ev if e["event"] == "joined"}
    assert sorted(joined.values()) == [1, 2]
    by = {}
    for e in ev:
        if e["event"] == "applied":
            by[e["worker"]] = by.get(e["worker"], 0) + 1
    assert set(by) == set(joined)                     # both workers' contributions arrived
    assert (run / "DONE").exists()
    res = json.loads(outs[0][0].strip().splitlines()[-1])
    assert res["val_bpb"] is not None and res["val_bpb"] < 4.0
    r1 = json.loads(outs[1][0].strip().splitlines()[-1])   # rank 1 of the 2-rank job: same steps, same lrs
    assert r1["rank"] == 1 and r1["steps"] == res["steps"] and r1["lr_sum"] == res["lr_sum"] > 0   # (progress_est)


def test_stalled_worker_leaves_the_round(tmp_path):
    """A worker whose heartbeat is fresh but which stopped publishing (a hung job) must stop inflating the round
    size, or every live worker's contribution would be scaled down by it."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    pool = Pool(tmp_path / "run", m, inner_steps=5, total_samples=10 ** 9, state_dtype=F64, worker_id="a",
                step_samples=4, stall_min_s=60, log=lambda *a: None)
    assert pool.round_samples() == 20
    now = time.time()
    pool.store.heartbeat("fresh", {"gpus": 1, "samples_per_round": 20, "joined": now})
    assert pool.round_samples() == 40
    pool.store.heartbeat("stuck", {"gpus": 1, "samples_per_round": 20, "joined": now - 900, "last_publish": now - 200,
                                   "round_s": 30.0})
    assert pool.round_samples() == 40                 # 200 s since its last publish > max(60, 3 x 30 s)
    assert len(pool.store.live_workers(180)) == 3     # ... but it still holds its GPUs against the cap
    pool.close()


def test_nothing_applies_after_done(tmp_path):
    from perf_kit.pool import Pool
    m = _tiny_model()
    pool = Pool(tmp_path / "run", m, inner_steps=1, total_samples=4, state_dtype=F64, worker_id="a",
                log=lambda *a: None)
    x, y = _batch(0)
    torch.nn.functional.mse_loss(m(x), y).backward()
    with torch.no_grad():
        for p in m.parameters():
            p.sub_(0.1 * p.grad)
    assert pool.sync(m, 4) and pool.done
    v = pool.store.latest()
    pool._publish(m)                                   # a straggler's late contribution
    assert pool._apply(wait=5) == 0 and pool.store.latest() == v
    pool.close()


def test_in_flight_round_stops_at_done(tmp_path):
    """When another worker finishes the run mid-round, this worker notices within done_check_every steps."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    pool = Pool(tmp_path / "run", m, inner_steps=100, total_samples=10 ** 9, state_dtype=F64, worker_id="a",
                done_check_every=5, log=lambda *a: None)
    for _ in range(4):
        assert not pool.sync(m, 4)
    pool.store.mark_done({"version": 0, "samples": 0})
    assert not pool.sync(m, 4) and pool.done and pool._steps == 5
    pool.close(m)
    assert not pool.store.pending()                    # nothing published after DONE


def test_sim_straggler_is_dropped_not_waited_for(tmp_path):
    """A worker 100x slower than two others: its contributions arrive more than max_staleness (4) rounds late and
    are dropped; nobody waits for it, it keeps going, and the run finishes."""
    run = tmp_path / "run"
    total = 12000
    procs = {"w0": _spawn(run, "w0", "--total_samples", total, "--slow", 0.002),
             "w1": _spawn(run, "w1", "--total_samples", total, "--slow", 0.002),
             "slow": _spawn(run, "slow", "--total_samples", total, "--slow", 0.2)}
    outs = _finish(procs)
    assert all(outs[w][0] == 0 for w in outs), {w: outs[w][2][-1500:] for w in outs}
    ev = _events(run)
    dropped = [e for e in ev if e["event"] == "dropped"]
    assert dropped and {e["worker"] for e in dropped} == {"slow"} and all(e["staleness"] > 4 for e in dropped)
    assert json.loads(outs["slow"][1].strip().splitlines()[-1])["contributions"] >= 2   # it kept going
    assert (run / "DONE").exists()
    assert json.loads(outs["w0"][1].strip().splitlines()[-1])["excess"] < 1e-3


def test_resume_after_everyone_left(tmp_path):
    """Every worker leaves before the run is done; a new worker on the same folder continues from the newest
    version with the samples, the momentum and the partial round intact (the folder is the checkpoint)."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    kw = dict(inner_steps=5, total_samples=10 ** 9, state_dtype=F64, log=lambda *a: None)
    a = Pool(tmp_path / "run", m, worker_id="a", **kw)
    for i in range(17):                                # 3 full rounds + 2 steps, published by close()
        x, y = _batch(i)
        opt.zero_grad()
        torch.nn.functional.mse_loss(m(x), y).backward()
        opt.step()
        a.sync(m, 4)
    a.close(m)
    st = Store(tmp_path / "run")
    before = st.load()
    assert before.samples == 68 and before.outer_steps == 3 and before.m is not None
    m2 = _tiny_model()                                 # a fresh job: its own init is overwritten by the pull
    b = Pool(tmp_path / "run", m2, worker_id="b", **kw)
    assert b.version == before.version and b.applied_samples == 68
    for n, p in m2.named_parameters():
        assert torch.equal(p.detach(), before.params[n])
    for _ in range(5):
        b.sync(m2, 4)                                  # no training: a zero pseudo-gradient
    after = st.load()
    assert after.samples == 88 and after.version == before.version + 1
    # a's partial round (8 of 20 samples) was kept in the outer state; b's zero contribution (weight 1) closes it:
    # m = 0.9 m_old + weighted mean of the round = 0.9 m_old + (buf_old + 0) / (acc_old + 1)
    assert abs(before.acc - 0.4) < 1e-12 and after.outer_steps == before.outer_steps + 1 and after.acc == 0.0
    for n in before.m:
        torch.testing.assert_close(after.m[n], 0.9 * before.m[n] + before.buf[n] / (before.acc + 1.0),
                                   rtol=0, atol=1e-12)
    b.close()


def test_claim_is_won_once(tmp_path):
    from perf_kit.pool import Pool
    m = _tiny_model()
    p1 = Pool(tmp_path / "run", m, inner_steps=5, total_samples=100, worker_id="a", log=lambda *a: None)
    p2 = Pool(tmp_path / "run", m, inner_steps=5, total_samples=100, worker_id="b", log=lambda *a: None)
    assert p1.claim("val_0003") and not p2.claim("val_0003") and p2.claim("val_0004")
    p1.close(); p2.close()


def test_progress_est_counts_the_live_round(tmp_path):
    """Until its first outer sync a worker's `progress` sees only its own samples (a progress-driven warm-up then
    runs the whole first round at a tiny lr); `progress_est` scales them by the pool's live round size."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    T = 10 ** 6
    pool = Pool(tmp_path / "run", m, inner_steps=5, total_samples=T, state_dtype=F64, worker_id="a",
                step_samples=4, done_check_every=2, log=lambda *a: None)
    assert pool.progress_est == pool.progress == 0.0
    now = time.time()                                  # two more jobs joined (1 and 2 GPUs), mid-round
    pool.store.heartbeat("b", {"gpus": 1, "samples_per_round": 20, "joined": now})
    pool.store.heartbeat("c", {"gpus": 2, "samples_per_round": 40, "joined": now})
    pool.store.heartbeat("d", {"gpus": 1, "samples_per_round": 0, "joined": now})   # joined without step_samples:
    pool.sync(m, 4)                                    # not counted. Step 1: b, c not seen yet (refreshed every
                                                       # done_check_every steps)
    assert pool.progress_est == pool.progress == 4 / T
    pool.sync(m, 4)                                    # step 2: refreshed; round = 20 + 20 + 40 = 4x own
    assert pool.progress == 8 / T and pool.progress_est == pytest.approx(8 * 4 / T)
    seen = []
    for _ in range(7):                                 # steps 3-9; step 5 publishes, applies (20 of 80), pulls
        pool.sync(m, 4)
        seen.append(pool.progress_est)
        assert pool.progress_est >= pool.progress
    assert pool.applied_samples == 20 and seen == sorted(seen)          # never decreases across the sync
    assert seen[2] == pytest.approx(80 / T)            # right after the sync: the end-of-round estimate holds
    assert pool.stats()["progress_est"] == pool.progress_est
    pool.close()


def test_progress_est_fixed_round_own_pace_and_solo(tmp_path):
    """With round_samples fixed in the config, and no step_samples given (own round size measured from the pace);
    for a pool of one, progress_est == progress."""
    from perf_kit.pool import Pool
    m = _tiny_model()
    T = 10 ** 6
    pool = Pool(tmp_path / "fixed", m, inner_steps=5, total_samples=T, round_samples=100, state_dtype=F64,
                worker_id="a", log=lambda *a: None)
    for k in range(1, 5):
        pool.sync(m, 4)
        assert pool.progress == 4 * k / T and pool.progress_est == pytest.approx(4 * k * (100 / 20) / T)
    pool.close()
    solo = Pool(tmp_path / "solo", m, inner_steps=5, total_samples=T, step_samples=4, state_dtype=F64,
                worker_id="s", done_check_every=1, log=lambda *a: None)
    for _ in range(12):
        solo.sync(m, 4)
        assert solo.progress_est == pytest.approx(solo.progress)
    solo.close()


# ------------------------------------------------------------------ pool2 fixes (2026-11-25)
def _bare_pool(run, cfg_extra=None):
    """A Pool object with only what _live_round needs (no join)."""
    from perf_kit.pool import Pool
    p = Pool.__new__(Pool)
    p.store = Store(run)
    p.cfg = {"heartbeat_timeout": 1e9, "stall_min_s": 300.0, **(cfg_extra or {})}
    return p


def test_first_round_worker_counts_in_round_size(tmp_path):
    """A live worker in its first round (joined 400 s ago, never published: compile + data start-up) counts in the
    round size for first_round_s; before pool2 it dropped out after stall_min_s (300 s) and the first arrivals of
    the 2026-11-25 VAE run were weighted 1.0, 0.5, 0.2, ... A worker that joined without step_samples beats 0."""
    now = time.time()
    st = Store(tmp_path)
    st.heartbeat("slow", {"gpus": 1, "joined": now - 400, "samples_per_round": 8000})
    st.heartbeat("fresh", {"gpus": 1, "joined": now - 10, "samples_per_round": 8000})
    st.heartbeat("nostep", {"gpus": 1, "joined": now - 400, "samples_per_round": 0})
    st.heartbeat("hung", {"gpus": 1, "joined": now - 5000, "samples_per_round": 8000})          # > first_round_s
    st.heartbeat("pub", {"gpus": 1, "joined": now - 2000, "samples_per_round": 8000, "last_publish": now - 200,
                         "round_s": 300})
    st.heartbeat("stalled", {"gpus": 1, "joined": now - 4000, "samples_per_round": 8000,
                             "last_publish": now - 2000, "round_s": 300})
    old = _bare_pool(tmp_path)                                       # a config written before pool2
    assert old._live_round() == 8000 + 8000                          # fresh + pub: the old behaviour
    new = _bare_pool(tmp_path, {"first_round_s": 3600.0})
    assert new._live_round() == 8000 + 8000 + 8000                   # + slow; nostep adds 0, hung / stalled out


def test_health_rule_fires_on_oscillation_not_on_noise():
    """Median of the last 5 pulled/local ratios > 1.25: fires on the as-run N=24 tiny-VAE sequence (oscillating,
    never 3 in a row above 1.5) and on the cluster run's proxy, silent on momentum-0 / synchronous / N=3."""
    from perf_kit.pool.worker import is_unhealthy
    asrun = [0.95, 1.0, 1.33, 1.14, 0.95, 1.37, 1.30, 0.97, 1.46, 1.51, 1.0, 1.46, 1.73, 1.16, 1.46]
    fired = [i for i in range(1, len(asrun) + 1) if is_unhealthy(asrun[:i])]
    assert fired and fired[0] == 7
    assert is_unhealthy([8.3, 4.6, 3.6])
    for ok in ([0.86] * 15, [0.92, 0.96, 0.9, 1.1, 0.95, 1.23, 1.05, 0.98], [1.0, 1.6, 1.0, 1.0, 1.0]):
        assert not any(is_unhealthy(ok[:i]) for i in range(1, len(ok) + 1))
