"""The call a training loop makes: `pool.sync(model, step_samples)` after every optimizer step.

    pool = Pool(run_dir, model, inner_steps=50, total_samples=10_000_000, max_gpus=16, gpus=world_size)
    if pool.full: sys.exit(0)                       # the pool already holds max_gpus; this job leaves quietly
    while not pool.done:
        loss = train_step(next_batch(pool.seed()))  # seed() changes every round and differs per worker
        lr = schedule(pool.progress_est) * min(1.0, (pool.local_steps + 1) / 200)  # pool progress AND local
                                                    # warm-up: each job's inner optimizer starts with empty moments
        if pool.sync(model, step_samples=global_batch):   # every inner_steps calls: publish, apply, pull
            ...                                     # weights were replaced by the newest global version
    pool.close(model)                               # publishes the partial round, leaves the pool

Inside a multi-GPU job (ordinary DDP), every rank calls the same methods; only rank 0 touches the files and the new
weights are broadcast from rank 0. Pooled: trainable parameters (through the outer optimizer, see outer.py) and
floating-point buffers (sample-weighted average). Not pooled, kept per worker: the inner optimizer state (as in
DiLoCo), AMP scaler, any EMA. A joining worker starts its inner optimizer fresh.

Schedules. `progress` is what the pool has APPLIED (at this worker's last pull) plus this worker's own unapplied
samples: until a worker's first outer sync the other workers' unapplied rounds are invisible to it, so a schedule
driven by `progress` runs the first round of a fresh run as if the pool were this one job (in a real run of 18 jobs
with a progress-based warm-up, the first 500 local steps trained at ~3 % of the nominal learning rate).
`progress_est` scales this worker's unapplied samples by the pool's live round size over its own samples per round
(the round size of the other workers publishing on time is refreshed at every pull and every `done_check_every`
steps, from their heartbeats): an estimate of the samples the whole pool has trained. It is never below `progress`
and never decreases. Drive learning-rate schedules by `progress_est`, and warm up by local steps as well
(`local_steps`): a joining job's inner optimizer starts with empty moments. `done` depends only on applied samples.
Pass `step_samples`: a worker that joined without it beats a round size of 0 until its first publish, so the other
workers cannot count it during its first round, which is exactly the round progress_est is for.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from dataclasses import asdict

import torch

from . import outer as O
from .store import LockBusy, Store

_STRIP = ("module.", "_orig_mod.")


def _clean(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for s in _STRIP:
            if name.startswith(s):
                name, changed = name[len(s):], True
    return name.replace("._orig_mod.", ".").replace(".module.", ".")


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "_orig_mod", model)


def weights_hash(tensors: dict) -> str:
    """sha256 over the named tensors' values (sorted names, float32 bytes on the CPU): identical weights, same hash."""
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().to("cpu", torch.float32).contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def unhealthy_median(ratios, window: int = 5) -> float:
    r = sorted(ratios[-window:])
    if not r:
        return 0.0
    n = len(r)
    return r[n // 2] if n % 2 else 0.5 * (r[n // 2 - 1] + r[n // 2])


def is_unhealthy(ratios, warn: float = 1.25, window: int = 5, min_n: int = 3) -> bool:
    """Health rule: the median pulled/local probe ratio of the last `window` syncs (at least `min_n`) > warn."""
    return len(ratios) >= min_n and unhealthy_median(ratios, window) > warn


def _dist():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist
    return None


def default_worker_id() -> str:
    job = os.environ.get("SLURM_JOB_ID")
    base = f"j{job}" if job else f"{socket.gethostname()}-{os.getpid()}"
    return f"{base}-{os.urandom(2).hex()}"     # a requeued job gets a new id: its old seqs stay valid


class Pool:
    def __init__(self, run_dir, model, *, inner_steps: int, total_samples: int, gpus: int | None = None,
                 max_gpus: int = 16, outer_lr: float = 0.7, outer_momentum: float = 0.9, mode: str = "dn",
                 max_staleness: float = 4.0, staleness_weight: str = "none", round_samples: float | None = None,
                 seed: int = 0, keep_versions: int = 3, keep_every: int = 0, heartbeat_s: float = 30.0,
                 heartbeat_timeout: float = 180.0, stall_min_s: float = 300.0, lock_timeout: float = 600.0, state_dtype=torch.float32,
                 worker_id: str | None = None, delta_dtype=None, step_samples: int | None = None,
                 done_check_every: int = 50, log=print, first_round_s: float = 3600.0, probe=None,
                 health_warn: float = 1.25):
        """Join (or create) the run in `run_dir`. Settings other than `gpus` / `worker_id` are fixed by whichever
        worker creates the run (config.json); later workers' values are ignored with a warning if different.
        `step_samples` (samples per optimizer step of this job), if given, lets the pool count this worker in the
        round size from the moment it joins rather than from its first contribution.

        `first_round_s` (pool2 fix, 2026-11-25): a live worker that has not published yet counts in the round size
        for this long after joining (before: only stall_min_s = 300 s, so on the VAE, whose first round took ~800 s
        with torch.compile, the first contributions were weighted 1.0, 0.5, 0.2, ... instead of 1/12 each).

        `probe(model) -> float` (pool2 health check; called on rank 0 only, must not use collectives): a loss on a
        fixed small batch. At every sync it is measured on the local weights just before publishing and on the
        pulled global weights; both and their ratio go to <run>/health/<worker>.jsonl and `self.health`. When the
        median ratio of the last 5 syncs (at least 3) exceeds `health_warn` it is logged loudly and written to
        events.jsonl once ("unhealthy"): the pooled weights are worse than what the workers train to (the 2026-11-25
        failure: ratio ~5 on the cluster; it OSCILLATES, so a "k in a row" rule can miss it - sim_pool_vae.py)."""
        d = _dist()
        self.rank = d.get_rank() if d else 0
        self.world = d.get_world_size() if d else 1
        self.gpus = int(gpus if gpus is not None else self.world)
        self.log = log if self.rank == 0 else (lambda *a, **k: None)
        self.worker = worker_id or default_worker_id()
        if d:                                   # every rank must agree on rank 0's id
            obj = [self.worker]
            d.broadcast_object_list(obj, src=0)
            self.worker = obj[0]
        self.delta_dtype = delta_dtype
        self.done_check_every = int(done_check_every)
        self._spr0 = int(step_samples) * int(inner_steps) if step_samples else 0
        self._spr = self._spr0                  # this worker's samples per round (every rank; set by a full round)
        self._others = 0.0                      # samples per round of the OTHER live workers (every rank)
        self._est_floor = 0.0                   # progress_est at the end of the last round (keeps it monotone)
        self.local_steps = 0                    # optimizer steps this job has taken (all rounds; local warm-up)
        self.state_dtype = state_dtype
        self._steps = 0
        self._samples = 0
        self._round = 0
        self._seq = 0
        self.full = False
        self.done = False
        self.changed_total = 0
        self.timing = {"publish_s": 0.0, "apply_s": 0.0, "pull_s": 0.0, "broadcast_s": 0.0}
        self.probe = probe
        self.health_warn = float(health_warn)
        self.health = {}
        self._ratios = []                       # pulled / local probe ratios of this worker's syncs
        self._warned_unhealthy = False
        self.pulled_hash = None                 # sha256 of the weights after the first pull (rank 0)
        if outer_momentum > 0 and mode == "dn":
            self.log("[pool] note: outer momentum > 0 with asynchronous workers diverged on the 2026-11-25 VAE "
                     "(24 workers, each re-converging within a round); momentum 0 is the safe setting there. "
                     "Watch the health ratio (probe=...).")
        want = dict(inner_steps=int(inner_steps), total_samples=int(total_samples), max_gpus=int(max_gpus),
                    outer=dict(lr=outer_lr, momentum=outer_momentum, mode=mode, max_staleness=max_staleness,
                               staleness_weight=staleness_weight),
                    round_samples=round_samples, seed=int(seed), keep_versions=int(keep_versions),
                    keep_every=int(keep_every), heartbeat_timeout=float(heartbeat_timeout),
                    stall_min_s=float(stall_min_s), first_round_s=float(first_round_s))
        info = None
        if self.rank == 0:
            self.store = Store(run_dir, lock_timeout=lock_timeout)
            info = self._join(model, want, heartbeat_s)
        if d:
            obj = [info]
            d.broadcast_object_list(obj, src=0)
            info = obj[0]
        self.cfg, self.index, self.full, self.done = info["cfg"], info["index"], info["full"], info["done"]
        self.ocfg = O.OuterConfig(**self.cfg["outer"])
        self.inner_steps = self.cfg["inner_steps"]
        self.total_samples = self.cfg["total_samples"]
        self.version, self.base_rounds, self.applied_samples = -1, 0.0, 0
        if not self.full and not self.done:
            self._pull(model)

    # ---------------------------------------------------------------- joining / leaving
    def _join(self, model, want, heartbeat_s):
        st = self.store
        with st.lock():
            cfg = st.init_config(want)
            for k, v in want.items():
                if cfg.get(k) != v:
                    self.log(f"[pool] note: run keeps {k}={cfg.get(k)!r} (this worker asked {v!r})")
            if st.latest() is None:
                st.commit(self._initial_state(model))
                st.event({"event": "created", "worker": self.worker})
            done = st.done()
            live = st.live_workers(cfg["heartbeat_timeout"])
            used = sum(w.get("gpus", 0) for w in live)
            full = (not done) and used + self.gpus > cfg["max_gpus"]
            index = -1
            if not full and not done:
                index = st.next_index()
                self._beat_info = {"gpus": self.gpus, "index": index, "host": socket.gethostname(),
                                   "samples_per_round": self._spr0, "joined": time.time()}
                st.heartbeat(self.worker, self._beat_info)
                st.event({"event": "joined", "worker": self.worker, "gpus": self.gpus, "index": index,
                          "live_gpus_before": used})
            else:
                st.event({"event": "refused", "worker": self.worker, "gpus": self.gpus, "live_gpus": used,
                          "done": done})
        if not full and not done:
            self._stop = threading.Event()
            self._hb = threading.Thread(target=self._beat_loop, args=(heartbeat_s,), daemon=True)
            self._hb.start()
        return {"cfg": cfg, "index": index, "full": full, "done": done}

    def _beat_loop(self, period):
        while not self._stop.wait(period):
            try:
                self.store.heartbeat(self.worker, self._beat_info)
            except OSError:
                pass

    def _initial_state(self, model) -> O.OuterState:
        params, buffers = self._local_tensors(model)
        return O.OuterState(params={k: v.to(self.state_dtype).clone() for k, v in params.items()},
                            buffers={k: v.to(self.state_dtype).clone() for k, v in buffers.items()})

    @staticmethod
    def _local_tensors(model):
        m = _unwrap(model)
        params = {_clean(n): p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
        buffers = {_clean(n): b.detach().cpu() for n, b in m.named_buffers() if b is not None
                   and torch.is_floating_point(b)}
        return params, buffers

    def close(self, model=None, publish_partial: bool = True) -> None:
        """Leave the pool. With `model`, the steps done since the last sync are published first (weighted by their
        samples, so a partial round is worth what it trained on)."""
        d = _dist()
        if model is not None and publish_partial and self._steps > 0 and not self.done and not self.full:
            if self.rank == 0:
                self._publish(model)
                try:
                    self._apply(wait=30.0)
                except LockBusy:
                    pass
            self._seq += 1
            self._steps = self._samples = 0
        if d:
            d.barrier()
        if self.rank == 0 and hasattr(self, "_stop"):
            self._stop.set()
            self.store.leave(self.worker)
            try:
                with self.store.lock(wait=30.0):
                    self.store.event({"event": "left", "worker": self.worker, "done": self.store.done()})
            except LockBusy:
                pass

    # ---------------------------------------------------------------- per-step API
    @property
    def progress(self) -> float:
        """Applied samples (at the last pull) plus this worker's unapplied ones, over the run's total. Exact but
        blind to the other workers' unapplied rounds: for schedules use `progress_est` (see the module docstring)."""
        return min(1.0, (self.applied_samples + self._samples) / max(1, self.total_samples))

    @property
    def progress_est(self) -> float:
        """Estimated progress of the whole pool, for schedules: applied samples plus this worker's unapplied ones
        scaled by (pool round size / own samples per round), i.e. every live worker is assumed to be as far into
        its round as this one. At least `progress`, never decreasing, equal to `progress` for a pool of one."""
        return max(self._est_floor, self._est_now())

    def _est_now(self) -> float:
        own = self._spr or (self._samples * self.inner_steps / self._steps if self._steps else 0.0)
        un = float(self._samples)
        if un and own > 0:
            R = float(self.cfg["round_samples"]) if self.cfg.get("round_samples") else self._others + own
            un *= max(1.0, R / own)
        return min(1.0, (self.applied_samples + un) / max(1, self.total_samples))

    def seed(self) -> int:
        """Data seed for the current round: differs per run seed, worker and round, never repeats."""
        h = hashlib.blake2b(f"{self.cfg['seed']}|{self.worker}|{self._round}".encode(), digest_size=8)
        return int.from_bytes(h.digest(), "little") & ((1 << 62) - 1)

    def sync(self, model, step_samples: int) -> bool:
        """Call after every optimizer step with the samples that step trained on (whole job, all ranks). Every
        `inner_steps` calls: publish the pseudo-gradient, apply what is in the inbox, load the newest global
        weights. Returns True when the model's weights were replaced."""
        if self.done or self.full:
            return False
        self._steps += 1
        self.local_steps += 1
        self._samples += int(step_samples)
        if self._steps < self.inner_steps:
            if self.done_check_every and self._steps % self.done_check_every == 0 and self._check_done():
                self.done = True                 # the run finished while this round was in flight: stop now
            return False
        return self.flush(model)

    def _check_done(self) -> bool:
        """Is the run DONE? Also refreshes the other workers' round size for progress_est (rank 0 reads the
        heartbeats, every rank receives the same value)."""
        obj = [self.store.done(), self._others_round()] if self.rank == 0 else [False, 0.0]
        d = _dist()
        if d:
            d.broadcast_object_list(obj, src=0)
        self._others = float(obj[1])
        return bool(obj[0])

    def flush(self, model) -> bool:
        """Publish / apply / pull now (every rank must call it)."""
        d = _dist()
        local_loss = None
        if self.rank == 0:
            if self.probe is not None:
                local_loss = float(self.probe(model))
            self._publish(model)
            try:
                self._apply(wait=10.0)
            except LockBusy:                     # another worker is applying; ours is in the inbox for the next
                pass
        self._seq += 1                           # counters advance on EVERY rank (only rank 0 wrote the file)
        if self._steps >= self.inner_steps:
            self._spr = self._samples            # a full round sets this worker's round size (as in _publish)
        self._est_floor = self.progress_est
        self._steps = 0
        self._samples = 0
        if d:
            d.barrier()
        self._pull(model)
        self._round += 1
        if self.rank == 0 and self.probe is not None:
            self._health(local_loss, float(self.probe(model)))
        return True

    def _health(self, local_loss, pulled_loss):
        """Pulled global weights vs this worker's own weights on the probe batch (rank 0)."""
        ratio = pulled_loss / max(local_loss, 1e-12)
        self.health = {"round": self._round, "version": self.version, "local": local_loss, "pulled": pulled_loss,
                       "ratio": ratio}
        hd = self.store.root / "health"
        hd.mkdir(exist_ok=True)
        with open(hd / f"{self.worker}.jsonl", "a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), **self.health}) + "\n")
        self._ratios.append(ratio)
        self.health["ratio_med5"] = unhealthy_median(self._ratios)
        if is_unhealthy(self._ratios, self.health_warn) and not self._warned_unhealthy:
            self._warned_unhealthy = True
            self.log(f"[pool] UNHEALTHY: over this worker's last syncs the pulled global weights' probe loss is "
                     f"{self.health['ratio_med5']:.2f}x its own (median): the outer step is not averaging well "
                     f"(outer {asdict(self.ocfg)}). Check outer momentum / lr.")
            try:
                with self.store.lock(wait=30.0):
                    self.store.event({"event": "unhealthy", "worker": self.worker, **self.health})
            except LockBusy:
                pass

    # ---------------------------------------------------------------- internals
    def _publish(self, model):
        t0 = time.time()
        params, buffers = self._local_tensors(model)
        g = self._global
        delta = {}
        for k, v in params.items():
            x = g.params[k] - v.to(g.params[k].dtype)
            delta[k] = x.to(self.delta_dtype) if self.delta_dtype else x
        for k, v in buffers.items():
            if k in g.buffers:
                delta[k] = g.buffers[k] - v.to(g.buffers[k].dtype)
        c = O.Contribution(worker=self.worker, seq=self._seq, samples=self._samples, base_rounds=g.rounds,
                           base_version=g.version, delta=delta)
        self.store.put(c)
        if self._steps >= self.inner_steps or not self._beat_info.get("samples_per_round"):
            self._beat_info["samples_per_round"] = self._samples    # a full round sets it; a partial only if unset
        now = time.time()
        if self._steps >= self.inner_steps:
            self._beat_info["round_s"] = now - self._beat_info.get("last_publish", self._beat_info["joined"])
        self._beat_info["last_publish"] = now
        self.store.heartbeat(self.worker, self._beat_info)
        self.timing["publish_s"] += time.time() - t0

    def _live_round(self, exclude: str | None = None) -> float:
        """Sum of samples per round over the live workers that publish on time (rank 0: reads the heartbeats)."""
        now = time.time()
        tot = 0
        for w in self.store.live_workers(self.cfg["heartbeat_timeout"]):
            if exclude is not None and w.get("worker") == exclude:
                continue
            # a heartbeat proves the process lives, not that it trains: a worker counts in the round only while
            # it publishes on time (3 of its own round times, at least stall_min_s, since its last publish).
            # pool2 fix: a worker still in its FIRST round (compile, data start-up) counts for first_round_s after
            # joining; before, it dropped out after stall_min_s and the first arrivals were over-weighted.
            last = w.get("last_publish")
            if last is None:
                ok = now - w.get("joined", now) <= self.cfg.get("first_round_s", self.cfg.get("stall_min_s", 300.0))
            else:
                ok = now - last <= max(self.cfg.get("stall_min_s", 300.0), 3.0 * w.get("round_s", 0.0))
            if ok:
                tot += w.get("samples_per_round", 0)
        return float(tot)

    def _others_round(self) -> float:
        return self._live_round(exclude=self.worker)

    def round_samples(self) -> float:
        if self.cfg.get("round_samples"):
            return float(self.cfg["round_samples"])
        tot = self._live_round()
        return tot if tot > 0 else float(self._beat_info.get("samples_per_round") or 1)

    def _apply(self, wait: float) -> int:
        """Fold every pending contribution into a new global version. Returns how many were consumed."""
        t0 = time.time()
        st = self.store
        n = 0
        with st.lock(wait=wait):
            files = st.pending()
            if not files or st.done():          # after DONE the final version stays final
                return 0
            state = st.load()
            contribs, ok_files = [], []
            for f in files:
                try:
                    contribs.append(st.read_contribution(f))
                    ok_files.append(f)
                except (FileNotFoundError, EOFError, RuntimeError):
                    pass
            R = self.round_samples()
            log = O.apply(state, contribs, R, self.ocfg)
            state.version += 1
            if st.commit(state):
                for f in ok_files:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
                for a in log:
                    st.event({"event": a.status, "worker": a.worker, "seq": a.seq, "staleness": round(a.staleness, 3),
                              "weight": round(a.weight, 4), "samples": a.samples, "version": state.version, "round_samples": R,
                              "applied_samples": state.samples, "by": self.worker})
                n = len(ok_files)
                st.prune(self.cfg["keep_versions"], self.cfg["keep_every"])
                if state.samples >= self.total_samples:
                    st.mark_done({"version": state.version, "samples": state.samples, "t": time.time()})
                    st.event({"event": "done", "version": state.version, "samples": state.samples})
        self.timing["apply_s"] += time.time() - t0
        return n

    def _pull(self, model):
        """Load the newest global version into the model (rank 0 reads, then broadcasts)."""
        t0 = time.time()
        d = _dist()
        m = _unwrap(model)
        named_p = {_clean(n): p for n, p in m.named_parameters() if p.requires_grad}
        named_b = {_clean(n): b for n, b in m.named_buffers() if b is not None and torch.is_floating_point(b)}
        meta = None
        if self.rank == 0:
            g = self.store.load(full=False)
            self._global = g
            if self.pulled_hash is None:         # pool2 check, first pull: names match exactly, nothing unpooled
                miss, extra = sorted(set(g.params) - set(named_p)), sorted(set(named_p) - set(g.params))
                if miss or extra:
                    raise RuntimeError(f"[pool] model and global weights differ in names: global-only {miss[:5]}, "
                                       f"model-only {extra[:5]} (compile / DDP wrapper or a different model?)")
                frozen = [_clean(n) for n, p in m.named_parameters() if not p.requires_grad]
                other = [_clean(n) for n, b in m.named_buffers() if b is not None and not torch.is_floating_point(b)]
                if frozen or other:
                    self.log(f"[pool] note: not pooled (kept per worker): frozen params {frozen[:5]}, "
                             f"non-float buffers {other[:5]}")
            with torch.no_grad():
                for k, p in named_p.items():
                    p.copy_(g.params[k].to(p.dtype))
                for k, b in named_b.items():
                    if k in g.buffers:
                        b.copy_(g.buffers[k].to(b.dtype))
            if self.pulled_hash is None:         # the copy is exact: weights after the pull == the global version
                mine = weights_hash({k: p for k, p in named_p.items()})
                ref = weights_hash({k: g.params[k].to(p.dtype) for k, p in named_p.items()})
                if mine != ref:
                    raise RuntimeError("[pool] weights after the first pull differ from the global version")
                self.pulled_hash = mine
                self.store.event({"event": "first_pull", "worker": self.worker, "version": g.version,
                                  "sha256": mine})
            meta = {"version": g.version, "rounds": g.rounds, "samples": g.samples, "done": self.store.done(),
                    "others": self._others_round()}
        self.timing["pull_s"] += time.time() - t0
        if d:
            t1 = time.time()
            obj = [meta]
            d.broadcast_object_list(obj, src=0)
            meta = obj[0]
            with torch.no_grad():
                for p in list(named_p.values()) + list(named_b.values()):
                    d.broadcast(p.data, src=0)
            self.timing["broadcast_s"] += time.time() - t1
        if meta["version"] != self.version:
            self.changed_total += 1
        self.version, self.base_rounds, self.applied_samples = meta["version"], meta["rounds"], meta["samples"]
        self._others = float(meta["others"])
        self.done = bool(meta["done"])

    def load_global(self, model, version: int | None = None, lookahead: bool = False) -> int:
        """Copy a global version (default newest) into `model` (rank 0 only; for final evaluation / export).
        Default: the heavy-ball iterate (the stored weights without the outer Nesterov look-ahead, see
        outer.export_params), which is what to evaluate or ship; lookahead=True gives the stored weights that
        workers train from."""
        g = self.store.load(version, full=not lookahead)
        params = g.params if lookahead else O.export_params(g, self.ocfg)
        m = _unwrap(model)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if _clean(n) in params:
                    p.copy_(params[_clean(n)].to(p.dtype))
            for n, b in m.named_buffers():
                if b is not None and _clean(n) in g.buffers:
                    b.copy_(g.buffers[_clean(n)].to(b.dtype))
        return g.version

    def claim(self, name: str) -> bool:
        """True for exactly one caller per `name` over the whole run (an O_EXCL file under <run>/claims). For work
        only one job should do, e.g. `if pool.claim(f"val_{pool.version // 20}"): validate()` - unlike a fixed job
        index, it survives the job that would have done it leaving. Every rank must call it."""
        ok = False
        if self.rank == 0:
            d = self.store.root / "claims"
            d.mkdir(exist_ok=True)
            try:
                os.close(os.open(d / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                ok = True
            except FileExistsError:
                ok = False
        dd = _dist()
        if dd:
            obj = [ok]
            dd.broadcast_object_list(obj, src=0)
            ok = obj[0]
        return bool(ok)

    def stats(self) -> dict:
        return {"worker": self.worker, "index": self.index, "gpus": self.gpus, "version": self.version,
                "applied_samples": self.applied_samples, "progress": self.progress,
                "progress_est": self.progress_est, "rounds_done": self._round, "contributions": self._seq,
                "timing": dict(self.timing), "outer": asdict(self.ocfg), "health": dict(self.health)}
