"""ElasticTCP: synchronous data-parallel training across SEPARATE Slurm jobs, which join and leave at any time.

    el = ElasticTCP(model, optimizer, run_dir, total_samples=N, micro_batch=b, global_micro=12)
    while not el.should_stop():
        with el.guard():                                  # a dead / hung peer anywhere in here -> regroup, step redone
            mis = el.micro_indices()                      # this rank's global micro-batch indices for this step
            for j, mi in enumerate(mis):
                with el.accumulate(last=j == len(mis) - 1):   # DDP no_sync except on the rank's last micro-batch
                    loss = el.model(*batch(seed=mi))      # el.model: this epoch's DDP wrapper (rebuilt per epoch)
                    el.backward(loss)
            set_lr(optimizer, schedule(el.progress) * el.lr_scale)
            el.step()                                     # verify the sync, optimizer step, join / leave / stop

Method (published, adapted; see README/WRITEUP of this directory and the 2026-11-26 note):
  * Membership epochs with in-memory state transfer, as in Elastic Horovod (state.commit()/restore(), hosts-updated
    interrupts at step boundaries; Horovod 0.20 docs, from memory) and torchft (per-step quorum, live recovery from a
    healthy peer instead of a checkpoint; pytorch/torchft, 2024-25, from memory). torchrun's elastic agent, in
    contrast, restarts ALL workers from a checkpoint on any membership change (PyTorch elastic docs, from memory):
    too slow for jobs that come and go every few minutes.
  * Within an epoch: plain DDP (Li et al., VLDB 2020: gradient buckets all-reduced during backward), the bf16
    compression comm hook (torch.distributed.algorithms.ddp_comm_hooks.default_hooks.bf16_compress_hook), no_sync
    for gradient accumulation.
  * Batch / lr as the world changes: either a FIXED global batch (`global_micro` micro-batches per optimizer step,
    split over the ranks; the math is identical for every world size, only throughput changes, as AdaptDL/Pollux
    do when they keep the batch, Qiao et al. OSDI 2021, from memory), or a fixed per-GPU batch with the global
    batch growing with the world and `lr_scale` = sqrt (Adam: Malladi et al. NeurIPS 2022, square-root scaling) or
    linear (SGD: Goyal et al. 2017) in global batch / `ref_batch`. Schedules run on GLOBAL SAMPLES (`progress`).

Protocol (one optimizer step):
  1. backward on this rank's micro-batches; DDP all-reduces the gradients (the last micro-batch syncs).
  2. a small control all-reduce, waited with a timeout: [alive count, join pending, stop, superseded, rank 0's
     agreed value (agree()), loss per rank,
     leave per rank, parameter checksum per rank (every `chk_every` steps)]. NCCL runs collectives in order, so a
     correct alive count proves the gradient sync finished on every rank.
  3. only then optimizer.step() (the commit). A failure in 1-2 leaves weights and optimizer untouched: the step is
     discarded and redone by the next epoch on the same micro-batch indices.
  4. leave / join / stop / desync are agreed from the control vector, so every rank takes the same branch.
Failure: a dead peer surfaces in < 1 s (closed socket), a hung one at `timeout_s` (TORCH_NCCL_ASYNC_ERROR_HANDLING=2
+ TORCH_NCCL_BLOCKING_WAIT=1: the watchdog aborts the communicator, wait() raises, the process lives; measured on
Arrhenius 2026-11-26, jobs 5775347-49). Survivors abort the group, race for the next epoch's claim file, the winner
waits until every old member checked in or is dead (heartbeat older than `dead_s`), and everyone gets the state of
the survivor with the highest committed step.
"""
from __future__ import annotations

import gc
import json
import os
import re
import signal
import socket
import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from . import state as S
from .membership import Membership, free_port, iface_ip

_DIST_MSG = re.compile(r"nccl|gloo|connection|timed out|timeout|socket|closed|abort|processgroup|store|"
                       r"remote process|broken pipe|reset by peer|distbackend|collective", re.I)


def is_dist_error(e: BaseException) -> bool:
    """A failure of the collective layer (peer died / hung / network), as opposed to a bug in the training code."""
    if isinstance(e, (TimeoutError, ConnectionError)):
        return True
    de = getattr(dist, "DistError", None)
    if de is not None and isinstance(e, de):
        return True
    return isinstance(e, RuntimeError) and bool(_DIST_MSG.search(str(e)[:2000]))


class _Superseded(Exception):
    pass


class ElasticTCP:
    def __init__(self, model, optimizer, run_dir, *, total_samples: int, micro_batch: int,
                 global_micro: int | None = None, accum: int = 1, forward_module=None, extra_modules=None,
                 extra: dict | None = None, lr_rule: str = "sqrt", ref_batch: int | None = None,
                 bf16_hook: bool = True, clip_norm: float | None = None, timeout_s: float = 120.0,
                 hb_s: float = 5.0, dead_s: float = 20.0, gather_s: float = 60.0, found_wait_s: float = 10.0,
                 join_settle_s: float = 3.0, regroup_min_s: float = 30.0, join_wait_s: float = 3 * 3600.0,
                 check_every_s: float = 5.0, ckpt_every_s: float = 900.0, ckpt_keep: int = 2, log_every: int = 10,
                 chk_every: int = 50, max_world: int = 64, iface: str | None = None, device=None, init_from=None,
                 init_counters: dict | None = None, leave_at: float | None = None, warmup=None,
                 ddp_kwargs: dict | None = None, hold_timeout_s: float = 4 * 3600.0, stall_s: float | None = None,
                 log=print):
        if os.name == "nt":
            os.environ.setdefault("USE_LIBUV", "0")
        # the elastic mode needs "abort the communicator, keep the process" (train_env.sh sets 1 = tear down)
        if os.environ.get("ELASTIC_KEEP_NCCL_ENV") != "1":
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2"
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        self.model_raw, self.optimizer = model, optimizer
        self.forward_module = forward_module if forward_module is not None else model
        self.modules = {"model": model, **(extra_modules or {})}
        self.extra = extra if extra is not None else {}
        self.root = Path(run_dir)
        self.total_samples, self.micro_batch = int(total_samples), int(micro_batch)
        self.global_micro, self.accum = (int(global_micro) if global_micro else None), int(accum)
        self.lr_rule, self.ref_batch = lr_rule, ref_batch
        self.bf16_hook, self.clip_norm = bf16_hook, clip_norm
        self.timeout_s, self.hb_s, self.dead_s, self.gather_s = timeout_s, hb_s, dead_s, gather_s
        self.found_wait_s, self.join_settle_s, self.regroup_min_s = found_wait_s, join_settle_s, regroup_min_s
        self.join_wait_s, self.check_every_s, self.ckpt_every_s, self.ckpt_keep = (join_wait_s, check_every_s,
                                                                                   ckpt_every_s, ckpt_keep)
        self.log_every, self.chk_every, self.hold_timeout_s = log_every, chk_every, hold_timeout_s
        # a rank stuck inside an epoch for this long (no committed step, not in a hold) cannot be saved in-process
        # (a gloo DDP all-reduce with a vanished peer waits forever on Windows; a deadlocked data loader): it exits
        # with code 75 so the others go on without it and Slurm frees the GPU. NCCL's own watchdog fires first.
        self.stall_s = stall_s if stall_s is not None else 3 * timeout_s + gather_s
        self._last_progress = time.time()
        self._holding = False
        self.max_world = min(max_world, self.global_micro) if self.global_micro else max_world
        self.init_from, self.init_counters = init_from, init_counters
        self.ddp_kwargs = dict(ddp_kwargs or {})
        self.log = log
        env_leave = os.environ.get("ELASTIC_LEAVE_AT")
        self.leave_at = leave_at if leave_at is not None else (float(env_leave) if env_leave else None)

        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
        if device is None:
            device = torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.backend = "nccl" if self.device.type == "cuda" else "gloo"
        self.job = os.environ.get("ELASTIC_JOB") or os.environ.get("SLURM_JOB_ID") or f"p{os.getpid()}"
        job_size = int(os.environ.get("ELASTIC_JOB_SIZE") or os.environ.get("LOCAL_WORLD_SIZE")
                       or os.environ.get("SLURM_NTASKS") or 1)
        self.mid = f"{self.job}-{self.local_rank}-{os.getpid()}"
        self.ip = iface_ip(iface if iface is not None else os.environ.get("ELASTIC_IFACE", "hsn0"))

        # counters (identical on every rank of an epoch)
        self.step_n, self.samples, self.micro = 0, 0, 0
        self.epoch, self.rank, self.world = -1, 0, 0
        self.roster_rec = None
        self.model = None                                   # this epoch's DDP wrapper
        self._finished = self._left = self._gave_up = False
        self._want_leave = False
        self._flags = {"join": False, "stop": False, "superseded": False}
        self._loss_sum, self._n_micro = None, 0
        self._agree_in, self.agreed = 0.0, 0.0
        self._regroup_t0 = None
        self._regroup_reason = "start"
        self._last_form = 0.0
        self._last_ckpt = time.time()
        self._ckpt_thread = None
        self._hold_n = 0
        self.timing = {"sync_s": 0.0, "steps": 0, "step_wall_s": 0.0, "regroups": 0, "failures": 0}
        self._t_last_step = None
        self._log_acc = {"n": 0, "sync": 0.0, "wall": 0.0, "wn": 0}
        self.history = []                                   # rank 0: the metrics rows written

        if warmup is not None:                              # compile / cudnn / data warm-up BEFORE announcing
            warmup()
            self.optimizer.zero_grad(set_to_none=True)
        self.mem = Membership(self.root, self.mid, {"ip": self.ip, "job": self.job, "local_rank": self.local_rank,
                                                    "job_size": job_size, "host": socket.gethostname(),
                                                    "pid": os.getpid(), "joined": time.time()}, hb_s=hb_s)
        self.mem.start_heartbeat()
        self._config()
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_want_leave", True))
        except (ValueError, OSError, AttributeError):
            pass
        self._watch_stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="elastic-watch", daemon=True)
        self._watcher.start()
        self._regroup_t0 = time.time()
        self._establish("join")

    # ================================================================ public
    @property
    def G(self) -> int:
        """Micro-batches per optimizer step over the whole world."""
        return self.global_micro if self.global_micro else max(1, self.world) * self.accum

    @property
    def global_batch(self) -> int:
        return self.G * self.micro_batch

    @property
    def progress(self) -> float:
        return self.samples / max(1, self.total_samples)

    @property
    def lr_scale(self) -> float:
        if self.global_micro or not self.ref_batch or self.lr_rule == "none":
            return 1.0
        r = self.global_batch / float(self.ref_batch)
        return r ** 0.5 if self.lr_rule == "sqrt" else r

    def agree(self, value: float) -> None:
        """Rank 0's `value` travels with the next step's control all-reduce; afterwards `self.agreed` holds it on
        every rank (e.g. a time-based schedule fraction, which must not differ between ranks)."""
        self._agree_in = float(value)

    def should_stop(self) -> bool:
        return self._finished or self._left or self._gave_up

    def micro_indices(self) -> range:
        """This rank's global micro-batch indices for the current step: a contiguous block of the step's G."""
        q, rem = divmod(self.G, self.world)
        start = self.rank * q + min(self.rank, rem)
        n = q + (1 if self.rank < rem else 0)
        return range(self.micro + start, self.micro + start + n)

    @contextmanager
    def accumulate(self, last: bool):
        if last or self.model is None or self.world <= 1:
            yield
        else:
            with self.model.no_sync():
                yield

    def backward(self, loss):
        """loss = mean over this micro-batch. Scaled so that DDP's average over ranks gives the mean over all G."""
        (loss * (self.world / self.G)).backward()
        lv = loss.detach().float()
        self._loss_sum = lv if self._loss_sum is None else self._loss_sum + lv
        self._n_micro += 1

    @contextmanager
    def guard(self):
        try:
            yield
        except Exception as e:                              # noqa: BLE001
            if not is_dist_error(e) or self.should_stop():
                raise
            self._failure(f"{type(e).__name__}: {str(e)[:300]}")

    def step(self) -> bool:
        """Verify the gradient sync, commit the optimizer step, then act on join / leave / stop. False: the step
        failed and was discarded (a regroup already happened)."""
        if self.should_stop():
            return False
        W, r = self.world, self.rank
        n = 5 + 3 * W
        c = torch.zeros(n, dtype=torch.float64, device=self.device)
        c[0] = 1.0
        if r == 0:
            c[1] = float(self._flags["join"] and W < self.max_world
                         and time.time() - self._last_form >= self.regroup_min_s)
            c[2] = float(self._flags["stop"])
            c[3] = float(self._flags["superseded"])
            c[4] = float(self._agree_in)                    # rank 0's value, the same on every rank after
        if self._loss_sum is not None:
            c[5 + r] = (self._loss_sum / max(1, self._n_micro)).to(c.dtype)
        c[5 + W + r] = float(self._leaving())
        chk = self.chk_every and self.step_n % self.chk_every == 0
        if chk:
            c[5 + 2 * W + r] = S.quick_checksum(self.modules).to(c.dtype)
        t0 = time.perf_counter()
        try:
            work = dist.all_reduce(c, async_op=True)
            work.wait(timeout=timedelta(seconds=self.timeout_s))
            v = c.tolist()
        except Exception as e:                              # noqa: BLE001
            if not is_dist_error(e):
                raise
            self._failure(f"control: {type(e).__name__}: {str(e)[:300]}")
            return False
        t_sync = time.perf_counter() - t0
        if round(v[0]) != W:
            self._failure(f"control: alive {v[0]} != world {W}")
            return False
        # ---- commit
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model_raw.parameters(), self.clip_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        G = self.G
        self.step_n += 1
        self._last_progress = time.time()
        self.samples += G * self.micro_batch
        self.micro += G
        self._loss_sum, self._n_micro = None, 0
        now = time.time()
        wall = now - self._t_last_step if self._t_last_step else 0.0
        self._t_last_step = now
        self.timing["sync_s"] += t_sync
        self.timing["steps"] += 1
        self.timing["step_wall_s"] += wall
        losses = v[5:5 + W]
        self.agreed = v[4]
        q, rem = divmod(G, W)
        counts = [q + (1 if i < rem else 0) for i in range(W)]
        self.last_loss = sum(l * k for l, k in zip(losses, counts)) / G
        self.last_losses = losses
        desync = False
        if chk:
            ck = v[5 + 2 * W:5 + 3 * W]
            desync = max(ck) != min(ck)
            if desync and r == 0:
                self.log(f"[elastic] DESYNC at step {self.step_n}: parameter checksums differ {ck}; re-broadcasting")
                self.mem.event({"event": "desync", "epoch": self.epoch, "step": self.step_n, "checksums": ck})
        self._metrics(losses, t_sync, wall, chk and not desync)
        leavers = [i for i in range(W) if v[5 + W + i] > 0.5]
        if v[3] > 0.5:                                      # another group took over the run (split brain guard)
            self._teardown(crashed=False)
            self._regroup_t0 = time.time()
            self._establish("join")
            return True
        if v[2] > 0.5 or self.samples >= self.total_samples:
            self._finish("stop file" if v[2] > 0.5 else "total samples reached")
            return True
        if leavers or v[1] > 0.5 or desync:
            self._planned(leavers, admit=v[1] > 0.5, reason=("leave " if leavers else "") +
                          ("join" if v[1] > 0.5 else "") + (" desync" if desync else ""))
            return True
        if self.ckpt_every_s and time.time() - self._last_ckpt >= self.ckpt_every_s:
            self._checkpoint(sync=False)
        return True

    @contextmanager
    def hold(self, tag: str = "hold"):
        """Every rank enters at the same step. Long one-rank work (eval on rank 0, a big save) goes inside; the other
        ranks wait on the epoch's TCPStore, not in a collective, so no collective timeout fires."""
        key = f"elastic_hold/{self.epoch}/{self.step_n}/{tag}/{self._hold_n}"
        self._hold_n += 1
        self._holding = True
        try:
            yield
            self._hold_sync(key)
        finally:
            self._holding = False
            self._last_progress = time.time()

    def state_dict(self) -> dict:
        return S.cpu_state(self.modules, self.optimizer, self._counters(), self.extra)

    def close(self):
        """Leave cleanly (all ranks); call when the loop ends for a reason of your own."""
        if not (self._finished or self._left):
            self._teardown(crashed=False)
            self._left = True
        self._watch_stop.set()
        self.mem.leave()

    # ================================================================ membership
    def _counters(self) -> dict:
        return {"step": self.step_n, "samples": self.samples, "micro": self.micro, "agreed": self.agreed}

    def _config(self):
        p = self.root / "config.json"
        want = {"total_samples": self.total_samples, "micro_batch": self.micro_batch,
                "global_micro": self.global_micro, "accum": self.accum, "lr_rule": self.lr_rule,
                "ref_batch": self.ref_batch, "bf16_hook": self.bf16_hook}
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps(want, indent=1).encode())
            os.close(fd)
        except FileExistsError:
            for _ in range(50):
                try:
                    have = json.loads(p.read_text())
                    break
                except (json.JSONDecodeError, PermissionError):
                    time.sleep(0.1)
            diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
            if diff:
                raise RuntimeError(f"[elastic] this job's settings differ from the run's config.json: {diff}")

    def _leaving(self) -> bool:
        if self.leave_at is not None and time.time() >= self.leave_at:
            self._want_leave = True
        return self._want_leave

    def _watch(self):
        while not self._watch_stop.wait(self.check_every_s):
            if (self.model is not None and not self._holding and self.stall_s
                    and time.time() - self._last_progress > self.stall_s):
                self.log(f"[elastic {self.mid}] STALLED: no step committed for {time.time() - self._last_progress:.0f}"
                         f" s (> stall_s {self.stall_s:.0f}) in epoch {self.epoch}: exiting (75) so the run goes on")
                try:
                    self.mem.event({"event": "stall_exit", "mid": self.mid, "epoch": self.epoch,
                                    "step": self.step_n})
                    self.mem.leave()
                finally:
                    os._exit(75)
            try:
                if self.mem.leave_requested(self.job):
                    self._want_leave = True
                if self.rank == 0 and self.epoch >= 0 and self.model is not None:
                    me = {m["mid"] for m in (self.roster_rec or {}).get("members", [])}
                    self._flags["join"] = bool(self.mem.complete_joiners(self.dead_s, self.join_settle_s,
                                                                         exclude=me))
                    self._flags["stop"] = self.mem.stop_requested()
                    n, rec = self.mem.latest_roster()
                    if n > self.epoch and rec and self.mid not in {m["mid"] for m in rec["members"]} \
                            and rec.get("leader") not in me:
                        self._flags["superseded"] = True
            except Exception:                               # noqa: BLE001
                pass

    def _establish(self, how: str, failed: int | None = None, reason: str = "", leavers=(), admit=False):
        """Loop until this process is in a formed epoch, or must stop. how: join | recover | planned."""
        for attempt in range(1000):
            n = None
            try:
                if how == "join":
                    got = self._join_roster()
                elif how == "recover":
                    got = self._recover_roster(failed, reason)
                else:
                    got = self._planned_roster(leavers, admit, reason)
                    how = "recover"                         # if forming fails, it is a failure of this epoch
                if got is None:
                    return False
                if got == "join":
                    how = "join"
                    continue
                n, rec = got
                self._form(n, rec)
                return True
            except _Superseded:
                how = "join"
                continue
            except Exception as e:                          # noqa: BLE001
                if not is_dist_error(e):
                    raise
                self.timing["failures"] += 1
                self._teardown(crashed=True)
                failed = n if n is not None else self.epoch
                reason = f"forming epoch {n}: {type(e).__name__}: {str(e)[:200]}"
                self.log(f"[elastic {self.mid}] {reason}")
                how = "recover"
        raise RuntimeError("[elastic] could not form an epoch after 1000 attempts")

    def _alive_in(self, n: int, rec: dict) -> list[dict]:
        ids = {m["mid"] for m in rec.get("members", [])}
        out = []
        for m in self.mem.members(fresh_s=3 * self.dead_s):
            if m["mid"] in ids and m.get("state") in ("active", "recovering") and m.get("epoch", -1) >= n - 1:
                out.append(m)
        return out

    def _join_roster(self):
        self.mem.update(state="joining", epoch=-1, step=self.step_n, ready=time.time())
        t0 = time.time()
        while True:
            if self.mem.done():
                self._finished = True
                self._leave_files()
                return None
            if self._leaving():
                self._left = True
                self._leave_files()
                return None
            n, rec = self.mem.latest_roster()
            if rec is not None and self.mid in {m["mid"] for m in rec["members"]} and n > self.epoch:
                return n, rec
            L = self.mem.latest_epoch()
            if L > n:                                       # an epoch is being formed (claimed, no roster yet)
                age = self.mem.claim_age(L)
                if age is not None and age > self.gather_s + 3 * self.timeout_s and self.mem.claim(L + 1):
                    return self._found(L + 1)
            elif rec is None or not self._alive_in(n, rec):
                if self.mem.claim(n + 1):                   # nobody runs: found (fresh run, or resume)
                    return self._found(n + 1)
            if time.time() - t0 > self.join_wait_s:
                self._gave_up = True
                self.log(f"[elastic {self.mid}] not admitted within {self.join_wait_s} s: leaving")
                self._leave_files()
                return None
            time.sleep(0.5)

    def _found(self, n: int):
        t0 = time.time()
        while time.time() - t0 < self.found_wait_s:         # let this job's other ranks / waiting jobs arrive
            j = self.mem.complete_joiners(3 * self.hb_s, 0.0, exclude={self.mid}, job_wait_s=self.found_wait_s)
            mine = [m for m in self.mem.members(3 * self.hb_s) if m.get("job") == self.job]
            if len(mine) >= int(self.mem.info["job_size"]) and time.time() - t0 > 1.0 and \
                    (j or time.time() - t0 > 0.5 * self.found_wait_s):
                break
            time.sleep(0.3)
        joiners = self.mem.complete_joiners(3 * self.hb_s, 0.0, exclude={self.mid}, job_wait_s=0.0)
        ck = S.latest_checkpoint(self.root / "ckpt")
        src_info = "own init"
        if ck is not None:
            d = S.load_checkpoint(ck, self.modules, self.optimizer)
            self._set_counters(d["counters"])
            self.extra.update(d.get("extra") or {})
            src_info = f"checkpoint {ck.name}"
        elif self.init_from is not None:
            src_info = self._load_init_from()
        me = self._member_rec(port=free_port(self.ip))
        members = [me] + [self._member_rec(m) for m in joiners][: self.max_world - 1]
        rec = {"epoch": n, "leader": self.mid, "members": members, "src_rank": 0, "step": self.step_n,
               "reason": f"found ({src_info})", "t": time.time()}
        self.mem.write_roster(n, rec)
        self.mem.event({"event": "found", "epoch": n, "mid": self.mid, "from": src_info, "world": len(members),
                        "step": self.step_n, "samples": self.samples})
        self._regroup_reason = rec["reason"]
        return n, rec

    def _load_init_from(self) -> str:
        f = self.init_from
        if callable(f):
            c = f(self.modules)
            if c:
                self._set_counters(c)
            return "init_from callable"
        f = str(f)
        if f.startswith("pool:"):
            from ..pool.store import Store
            from ..pool.worker import _clean
            g = Store(f[5:]).load(full=False)
            named = {_clean(k): p for k, p in self.model_raw.named_parameters()}
            miss = sorted(set(named) - set(g.params))
            if miss:
                raise RuntimeError(f"[elastic] pool weights lack {miss[:5]}")
            with torch.no_grad():
                for k, p in named.items():
                    p.copy_(g.params[k].to(p.dtype))
                for k, b in self.model_raw.named_buffers():
                    if _clean(k) in g.buffers:
                        b.copy_(g.buffers[_clean(k)].to(b.dtype))
            self._set_counters(self.init_counters or {})
            return f"pool {f[5:]} version {g.version}"
        d = torch.load(f, map_location="cpu", weights_only=False)
        sd = d.get("modules", {}).get("model") if isinstance(d, dict) and "modules" in d else d
        self.model_raw.load_state_dict(sd)
        self._set_counters(self.init_counters or {})
        return f"weights {f}"

    def _set_counters(self, c: dict):
        self.step_n = int(c.get("step", 0))
        self.samples = int(c.get("samples", 0))
        self.micro = int(c.get("micro", 0))
        self.agreed = float(c.get("agreed", 0.0))

    def _member_rec(self, m: dict | None = None, port: int | None = None) -> dict:
        if m is None:
            m = {**self.mem.info, "mid": self.mid}
        out = {k: m.get(k) for k in ("mid", "ip", "job", "local_rank", "host")}
        if port is not None:
            out["port"] = port
        return out

    def _recover_roster(self, failed: int, reason: str):
        """After a failure in epoch `failed`: check in, then either lead the next epoch or wait for its roster."""
        self.mem.update(state="recovering", epoch=failed, step=self.step_n)
        target = failed + 1
        t0 = time.time()
        while True:
            if self.mem.done():
                self._finished = True
                self._leave_files()
                return None
            rec = self.mem.roster(target)
            if rec is not None:
                if self.mid in {m["mid"] for m in rec["members"]}:
                    return target, rec
                return "join"                               # checked in too late: come back as a newcomer
            if self.mem.claim(target):
                return self._lead_recovery(target, failed, reason)
            age = self.mem.claim_age(target)
            if age is not None and age > self.gather_s + 3 * self.timeout_s:
                target += 1                                 # the claimer died before writing its roster
            if time.time() - t0 > 4 * (self.gather_s + self.timeout_s):
                return "join"
            time.sleep(0.3)

    def _lead_recovery(self, n: int, failed: int, reason: str):
        old = self.mem.roster(failed) or {"members": []}
        old_ids = [m["mid"] for m in old["members"]]
        t0 = time.time()
        while True:
            now = time.time()
            ms = {m["mid"]: m for m in self.mem.members()}
            checked, pending = [], []
            for mid in old_ids:
                m = ms.get(mid)
                if m is None or now - m.get("beat", 0) > self.dead_s or m.get("state") == "joining":
                    continue                                # dead or gone
                if m.get("state") == "recovering" and m.get("epoch") == failed:
                    checked.append(m)
                else:
                    pending.append(mid)
            if self.mid not in old_ids:
                checked.append({**self.mem.info, **self.mem.fields, "mid": self.mid})
            if not pending or now - t0 > self.gather_s:
                break
            time.sleep(0.3)
        survivors = sorted(checked, key=lambda m: (m["mid"] != self.mid,
                                                   old_ids.index(m["mid"]) if m["mid"] in old_ids else 1 << 30))
        have = {m["mid"] for m in survivors}
        joiners = [m for m in self.mem.complete_joiners(self.dead_s, 0.0, exclude=have)]
        members = [self._member_rec(survivors[0], port=free_port(self.ip))] + \
                  [self._member_rec(m) for m in survivors[1:]] + [self._member_rec(m) for m in joiners]
        members = members[: self.max_world]
        steps = [int(m.get("step", 0)) for m in survivors]
        src = max(range(len(survivors)), key=lambda i: (steps[i], -i))
        rec = {"epoch": n, "leader": self.mid, "members": members, "src_rank": src, "step": steps[src],
               "reason": f"recovery after epoch {failed}: {reason[:200]}", "t": time.time(),
               "lost": [m for m in old_ids if m not in have]}
        self.mem.write_roster(n, rec)
        self.mem.event({"event": "recovery", "epoch": n, "failed_epoch": failed, "leader": self.mid,
                        "world": len(members), "lost": rec["lost"], "gather_s": round(time.time() - t0, 2),
                        "src_step": steps[src], "steps": steps, "reason": reason[:300]})
        self._regroup_reason = rec["reason"]
        return n, rec

    def _planned_roster(self, leavers, admit: bool, reason: str):
        W = self.world
        survivors = [i for i in range(W) if i not in leavers]
        old = self.roster_rec
        n = self.epoch + 1
        if not survivors:                                   # every rank leaves: checkpoint, then go
            self._checkpoint(sync=True)
            self.mem.event({"event": "all_left", "epoch": self.epoch, "step": self.step_n, "samples": self.samples})
            self._teardown(crashed=False)
            self._left = True
            self._leave_files()
            return None
        leader = survivors[0]
        if self.rank == leader:
            ids = {m["mid"] for m in old["members"]}
            joiners = self.mem.complete_joiners(self.dead_s, self.join_settle_s, exclude=ids) if admit else []
            if not self.mem.claim(n):
                other = self.mem.roster(n)
                self._teardown(crashed=False)
                if other is not None and self.mid not in {m["mid"] for m in other["members"]}:
                    raise _Superseded()
                raise TimeoutError(f"epoch {n} claimed by someone else during a planned regroup")
            keep = [old["members"][i] for i in survivors]
            members = [dict(keep[0], port=free_port(self.ip))] + keep[1:] + [self._member_rec(m) for m in joiners]
            members = members[: self.max_world]
            rec = {"epoch": n, "leader": self.mid, "members": members, "src_rank": 0, "step": self.step_n,
                   "reason": reason.strip() or "planned", "t": time.time(),
                   "left": [old["members"][i]["mid"] for i in leavers], "joined": [m["mid"] for m in joiners]}
            self.mem.write_roster(n, rec)
            self.mem.event({"event": "regroup", "epoch": n, "world": len(members), "reason": rec["reason"],
                            "left": rec["left"], "joined": rec["joined"], "step": self.step_n})
        self._teardown(crashed=False)
        if self.rank in leavers:
            self._left = True
            self._leave_files()
            return None
        self.mem.update(state="recovering", epoch=self.epoch, step=self.step_n)
        t0 = time.time()
        while True:
            rec = self.mem.roster(n)
            if rec is not None:
                if self.mid in {m["mid"] for m in rec["members"]}:
                    self._regroup_reason = rec["reason"]
                    return n, rec
                return "join"
            if time.time() - t0 > self.gather_s + self.timeout_s:
                raise TimeoutError(f"no roster for epoch {n} after a planned regroup")
            time.sleep(0.2)

    def _form(self, n: int, rec: dict):
        ids = [m["mid"] for m in rec["members"]]
        rank, world = ids.index(self.mid), len(ids)
        m0 = rec["members"][0]
        t0 = time.time()
        kw = dict(init_method=f"tcp://{m0['ip']}:{m0['port']}", rank=rank, world_size=world,
                  timeout=timedelta(seconds=self.timeout_s))
        if self.backend == "nccl":
            kw["device_id"] = self.device
        dist.init_process_group(self.backend, **kw)
        self.epoch, self.rank, self.world, self.roster_rec = n, rank, world, rec
        t_init = time.time() - t0
        src = int(rec.get("src_rank", 0))
        got = S.broadcast_state(self.modules, self.optimizer, self._counters(), self.extra, src, self.device)
        self._set_counters(got["counters"])
        if rank != src:
            self.extra.clear()
            self.extra.update(got["extra"])
        t_sync = time.time() - t0 - t_init
        h = S.fingerprint(self.modules)
        hs = [None] * world
        dist.all_gather_object(hs, h)
        if any(x != hs[src] for x in hs):
            raise AssertionError(f"[elastic] parameter hash after the state transfer differs: {hs}")
        self._build_ddp()
        self.optimizer.zero_grad(set_to_none=True)
        self._loss_sum, self._n_micro = None, 0
        self._flags["join"] = self._flags["superseded"] = False
        self._last_form = time.time()
        self._last_progress = time.time()
        self._t_last_step = None
        self.timing["regroups"] += 1
        self.mem.update(state="active", epoch=n, step=self.step_n)
        regroup_s = time.time() - (self._regroup_t0 or t0)
        self._regroup_t0 = None
        if rank == 0:
            self.mem.event({"event": "epoch", "epoch": n, "world": world, "src_rank": src, "step": self.step_n,
                            "samples": self.samples, "reason": self._regroup_reason,
                            "hosts": [m.get("host") for m in rec["members"]], "jobs": [m.get("job") for m in
                                                                                      rec["members"]],
                            "init_s": round(t_init, 3), "state_sync_s": round(t_sync, 3),
                            "regroup_s": round(regroup_s, 3), "sha256": h[:16]})
            self.log(f"[elastic] epoch {n}: world {world}, step {self.step_n}, samples {self.samples}, "
                     f"regroup {regroup_s:.2f} s (init {t_init:.2f}, state {t_sync:.2f}) - {self._regroup_reason}")

    def _build_ddp(self):
        import inspect
        from torch.nn.parallel import DistributedDataParallel as DDP
        kw = dict(gradient_as_bucket_view=True, **self.ddp_kwargs)
        if self.device.type == "cuda":
            kw["device_ids"] = [self.device.index]
        if "init_sync" in inspect.signature(DDP.__init__).parameters:
            kw.setdefault("init_sync", False)                # the state was just broadcast and hash-checked
        self.model = DDP(self.forward_module, **kw)
        if self.bf16_hook and self.world > 1 and self.backend == "nccl":   # the hook needs NCCL (CPU: fp32)
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as H
            self.model.register_comm_hook(state=None, hook=H.bf16_compress_hook)

    def _teardown(self, crashed: bool):
        if self.model is not None:
            self.model = None
            gc.collect()
        if dist.is_available() and dist.is_initialized():
            if crashed and self.backend == "nccl" and hasattr(dist.distributed_c10d, "_abort_process_group"):
                try:
                    dist.distributed_c10d._abort_process_group()
                except Exception:                           # noqa: BLE001
                    pass
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:                               # noqa: BLE001
                pass
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self._loss_sum, self._n_micro = None, 0

    def _failure(self, reason: str):
        self.timing["failures"] += 1
        failed = self.epoch
        self.log(f"[elastic {self.mid}] failure in epoch {failed} at step {self.step_n}: {reason[:300]}")
        self._teardown(crashed=True)
        self._regroup_t0 = time.time()
        self._establish("recover", failed=failed, reason=reason)

    def _planned(self, leavers, admit: bool, reason: str):
        self._regroup_t0 = time.time()
        if self.rank == 0 and self._ckpt_thread is not None:
            self._ckpt_thread.join()
        self._establish("planned", leavers=leavers, admit=admit, reason=reason)

    def _leave_files(self):
        self._watch_stop.set()
        self.mem.leave()

    def _finish(self, why: str):
        if self.rank == 0:
            path = self._checkpoint(sync=True, hold=False)
            from ..pool.store import write_json
            write_json(self.root / "DONE", {"why": why, "step": self.step_n, "samples": self.samples,
                                            "checkpoint": str(path), "t": time.time()})
            self.mem.event({"event": "done", "why": why, "step": self.step_n, "samples": self.samples})
        try:
            self._hold_sync(f"elastic_finish/{self.epoch}")
        except Exception as e:                              # noqa: BLE001
            if not is_dist_error(e):
                raise
        self._teardown(crashed=False)
        self._finished = True
        self._leave_files()

    # ================================================================ checkpoints, holds, metrics
    def _checkpoint(self, sync: bool, hold: bool = True):
        """Rank 0 saves <run>/ckpt/step_*.pt. sync=False: CPU copy now, file written by a thread."""
        path = None
        if self.rank == 0:
            if self._ckpt_thread is not None:
                self._ckpt_thread.join()
            st = S.cpu_state(self.modules, self.optimizer, self._counters(), self.extra)
            st["epoch"], st["world"] = self.epoch, self.world
            path = self.root / "ckpt" / f"step_{self.step_n:09d}.pt"

            def save():
                t0 = time.time()
                S.save_checkpoint(self.root / "ckpt", st, keep=self.ckpt_keep)
                self.mem.event({"event": "checkpoint", "step": st["counters"]["step"],
                                "samples": st["counters"]["samples"], "s": round(time.time() - t0, 2)})
            if sync:
                save()
            else:
                self._ckpt_thread = threading.Thread(target=save, daemon=False)
                self._ckpt_thread.start()
        self._last_ckpt = time.time()
        if sync and hold and self.world > 1:
            self._hold_sync(f"elastic_ckpt/{self.epoch}/{self.step_n}")
        return path

    def _hold_sync(self, key: str):
        if not (dist.is_available() and dist.is_initialized()) or self.world <= 1:
            return
        store = dist.distributed_c10d._get_default_store()
        if self.rank == 0:
            store.set(key, "1")
            t0 = time.time()                                # wait for the others' acks: rank 0 may tear the
            while time.time() - t0 < max(60.0, self.timeout_s):   # store down right after (finish, all-left)
                try:
                    if store.add(key + "/ack", 0) >= self.world - 1:
                        return
                except Exception:                           # noqa: BLE001
                    return
                time.sleep(0.1)
            return
        r0 = self.roster_rec["members"][0]["mid"]
        t0 = time.time()
        last_beat_check = 0.0
        while True:
            if store.check([key]):
                store.add(key + "/ack", 1)
                return
            now = time.time()
            if now - last_beat_check > self.hb_s:
                last_beat_check = now
                m = [x for x in self.mem.members() if x["mid"] == r0]
                if not m or now - m[0].get("beat", 0) > self.dead_s:
                    raise TimeoutError(f"hold {key}: rank 0 ({r0}) is gone")
            if now - t0 > self.hold_timeout_s:
                raise TimeoutError(f"hold {key}: waited {self.hold_timeout_s} s")
            time.sleep(0.2)

    def _metrics(self, losses, t_sync, wall, chk_ok):
        a = self._log_acc
        a["n"] += 1
        a["sync"] += t_sync
        if wall > 0:
            a["wall"] += wall
            a["wn"] = a.get("wn", 0) + 1
        if self.rank != 0 or self.step_n % self.log_every:
            return
        row = {"t": round(time.time(), 3), "step": self.step_n, "samples": self.samples, "epoch": self.epoch,
               "world": self.world, "loss": self.last_loss, "loss_ranks": [round(x, 6) for x in losses],
               "ctrl_wait_ms": round(1e3 * a["sync"] / a["n"], 3), "step_s": round(a["wall"] / max(1, a.get("wn", 0)), 5),
               "lr_scale": self.lr_scale, "global_batch": self.global_batch}
        if chk_ok:
            row["chk_equal"] = True
        self.history.append(row)
        with open(self.root / "metrics" / "elastic.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        self._log_acc = {"n": 0, "sync": 0.0, "wall": 0.0, "wn": 0}
