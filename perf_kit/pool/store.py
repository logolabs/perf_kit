"""The run folder on a shared filesystem (Lustre on the cluster, NTFS in the tests). No server, no NCCL between jobs.

    <run>/config.json            pool + outer settings, written once by the first worker
    <run>/global/v00000042.pt    one immutable file per global version: weights + counters (what workers read),
    <run>/global/v00000042.opt.<token>.pt   ... and its outer-optimizer state (momentum, partial round; applier only)
    <run>/inbox/c-<worker>-<seq>.pt   pseudo-gradients waiting to be applied
    <run>/workers/<worker>.json  heartbeat of each live worker (GPUs, samples per round, last beat)
    <run>/workers/_next          next worker index (a small integer per join, for data / shard offsets)
    <run>/events.jsonl           applied / dropped / joined / left, one JSON line each (written under the lock)
    <run>/lock                   O_EXCL lock file; taken over if older than `lock_timeout` seconds
    <run>/DONE                   written once `samples` reaches the run's total

Rules that keep it correct when jobs die at any instruction:
  * every file is written to a private temporary name and then moved to a NEW final name (never over an existing
    file: on Windows that fails while someone reads it, and on Lustre it would race);
  * a new global version is claimed by os.link(tmp, final), which fails if the name exists, so two appliers can
    never both commit version n+1 even if a stale lock was wrongly taken over;
  * an applier deletes inbox files only after its commit, and each version records per-worker high-water marks, so
    a crash between commit and delete cannot apply a contribution twice;
  * nothing ever waits on a particular worker: an unarrived contribution is simply never applied.
"""
from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                        # torch is imported lazily: the launcher runs on login nodes without it
    from .outer import Contribution, OuterState

_VER = re.compile(r"^v(\d{8})\.pt$")
_CON = re.compile(r"^c-(.+)-(\d{8})\.pt$")


def _token() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _publish(tmp: Path, final: Path, exclusive: bool) -> bool:
    """Move a finished temporary file to its final name. exclusive=True -> False if `final` already exists."""
    try:
        os.link(tmp, final)              # atomic create-if-absent on POSIX (Lustre included) and NTFS
        os.unlink(tmp)
        return True
    except FileExistsError:
        os.unlink(tmp)
        return False
    except OSError:                      # filesystem without hard links
        if exclusive and final.exists():
            os.unlink(tmp)
            return False
        os.rename(tmp, final)            # Windows: fails if final exists; POSIX: the lock is the guard
        return True


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(f".{path.name}.{_token()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    for _ in range(50):                  # replace over our OWN heartbeat file; a reader may hold it on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05)
    os.unlink(tmp)


def read_json(path: Path, default=None):
    for _ in range(20):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return default


class LockBusy(Exception):
    pass


class Store:
    def __init__(self, run_dir, lock_timeout: float = 600.0):
        self.root = Path(run_dir)
        self.lock_timeout = lock_timeout
        for d in ("global", "inbox", "workers"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- lock
    @contextmanager
    def lock(self, wait: float = 120.0, poll: float = 0.2):
        """Exclusive section. Raises LockBusy after `wait` seconds. A lock older than lock_timeout (its holder died)
        is moved aside by exactly one taker (rename of the same source succeeds once)."""
        path = self.root / "lock"
        me = _token()
        t0 = time.time()
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, json.dumps({"holder": me, "t": time.time()}).encode())
                os.close(fd)
                break
            except FileExistsError:
                try:
                    age = time.time() - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age > self.lock_timeout:
                    try:
                        aside = self.root / f".lock.stale.{me}"
                        os.rename(path, aside)
                        os.unlink(aside)
                        self.event({"event": "lock_takeover", "age_s": round(age, 1)}, locked=False)
                    except OSError:
                        pass
                    continue
                if time.time() - t0 > wait:
                    raise LockBusy(f"{path} held for {age:.0f} s")
                time.sleep(poll)
        try:
            yield
        finally:
            try:
                if (read_json(path, {}) or {}).get("holder") == me:
                    os.unlink(path)
            except OSError:
                pass

    def event(self, rec: dict, locked: bool = True) -> None:
        rec = {"t": round(time.time(), 3), **rec}
        with open(self.root / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------------------------------------------------------------- config
    def config(self):
        return read_json(self.root / "config.json")

    def init_config(self, cfg: dict) -> dict:
        """Write config.json if absent (call under the lock); return the stored one."""
        path = self.root / "config.json"
        if not path.exists():
            tmp = path.with_name(f".config.{_token()}.tmp")
            tmp.write_text(json.dumps(cfg, indent=1))
            _publish(tmp, path, exclusive=True)
        return self.config()

    # ---------------------------------------------------------------- versions
    def versions(self) -> list[int]:
        out = []
        for p in (self.root / "global").iterdir():
            m = _VER.match(p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def latest(self) -> int | None:
        v = self.versions()
        return v[-1] if v else None

    def _vpath(self, v: int) -> Path:
        return self.root / "global" / f"v{v:08d}.pt"

    _OPT = ("m", "buf", "acc", "high_water")       # only the applier needs these: kept out of what workers read

    def load(self, v: int | None = None, full: bool = True) -> OuterState | None:
        """Load version v (default: latest). full=False skips the outer-optimizer file (what a worker pulls).
        Retries if the latest version is pruned between listing and opening."""
        for _ in range(10):
            ver = self.latest() if v is None else v
            if ver is None:
                return None
            import torch
            from .outer import OuterState
            try:
                d = torch.load(self._vpath(ver), map_location="cpu", weights_only=False)
                opt_name = d.pop("opt_file")
                if full:
                    d.update(torch.load(self.root / "global" / opt_name, map_location="cpu", weights_only=False))
                else:
                    d.update(m=None, buf=None, acc=0.0, high_water={})
                return OuterState.from_dict(d)
            except (FileNotFoundError, PermissionError, EOFError, RuntimeError):
                if v is not None:
                    raise
                time.sleep(0.2)
        raise RuntimeError("could not load the latest global version")

    def commit(self, state: OuterState) -> bool:
        """Claim version state.version. False if someone else already committed that number. The outer-optimizer
        part goes to a uniquely named file first; the weights file (which names it) is the commit point."""
        import torch
        d = state.to_dict()
        tok = _token()
        opt_name = f"v{state.version:08d}.opt.{tok}.pt"
        torch.save({k: d.pop(k) for k in self._OPT}, self.root / "global" / opt_name)
        d["opt_file"] = opt_name
        final = self._vpath(state.version)
        tmp = final.with_name(f".{final.name}.{tok}.tmp")
        torch.save(d, tmp)
        ok = _publish(tmp, final, exclusive=True)
        if not ok:
            try:
                os.unlink(self.root / "global" / opt_name)
            except OSError:
                pass
        return ok

    def prune(self, keep: int, keep_every: int = 0) -> None:
        vs = self.versions()
        if not vs:
            return
        kept = set(vs[-keep:]) if keep > 0 else set(vs)
        for v in vs:
            if v in kept or (keep_every and v % keep_every == 0):
                continue
            try:
                os.unlink(self._vpath(v))
            except OSError:                  # Windows: a reader has it open; next prune gets it
                pass
        oldest = min(kept)
        for p in (self.root / "global").glob("v*.opt.*.pt"):     # outer state of pruned / never-committed versions
            v = int(p.name[1:9])
            if v < oldest and not (keep_every and v % keep_every == 0):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ---------------------------------------------------------------- inbox
    def put(self, c: Contribution) -> Path:
        import torch
        final = self.root / "inbox" / f"c-{c.worker}-{c.seq:08d}.pt"
        tmp = final.with_name(f".{final.name}.{_token()}.tmp")
        torch.save({"worker": c.worker, "seq": c.seq, "samples": c.samples, "base_rounds": c.base_rounds,
                    "base_version": c.base_version, "delta": c.delta}, tmp)
        _publish(tmp, final, exclusive=True)
        return final

    def pending(self) -> list[Path]:
        """Finished contribution files, oldest first (by file time, then name)."""
        out = []
        for p in (self.root / "inbox").iterdir():
            if _CON.match(p.name):
                try:
                    out.append((p.stat().st_mtime, p.name, p))
                except FileNotFoundError:
                    pass
        return [p for _, _, p in sorted(out)]

    @staticmethod
    def read_contribution(p: Path) -> Contribution:
        import torch
        from .outer import Contribution
        d = torch.load(p, map_location="cpu", weights_only=False)
        return Contribution(**d)

    # ---------------------------------------------------------------- workers
    def heartbeat(self, worker: str, info: dict) -> None:
        write_json(self.root / "workers" / f"{worker}.json", {**info, "worker": worker, "beat": time.time()})

    def leave(self, worker: str) -> None:
        try:
            os.unlink(self.root / "workers" / f"{worker}.json")
        except OSError:
            pass

    def live_workers(self, timeout: float) -> list[dict]:
        now = time.time()
        out = []
        for p in (self.root / "workers").glob("*.json"):
            d = read_json(p)
            if d and now - d.get("beat", 0) <= timeout and not d.get("left"):
                out.append(d)
        return out

    def next_index(self) -> int:
        """A small integer per join (call under the lock)."""
        path = self.root / "workers" / "_next"
        n = int((read_json(path, {"n": 0}) or {"n": 0})["n"])
        write_json(path, {"n": n + 1})
        return n

    def done(self) -> bool:
        return (self.root / "DONE").exists()

    def mark_done(self, info: dict) -> None:
        if not self.done():
            write_json(self.root / "DONE", info)
