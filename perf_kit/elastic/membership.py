"""The run folder's membership files: who is in the run, which epoch is current, who leads the next one.

    <run>/members/<mid>.json     one file per PROCESS (a job of 4 GPUs has 4): ip, a fresh port, job id, local rank,
                                 job size, state (joining | active | recovering), epoch, committed step, heartbeat.
                                 Rewritten every `hb_s` by a thread; deleted when the process leaves cleanly.
    <run>/epochs/e000042.claim   O_EXCL: whoever creates it leads epoch 42 (writes its roster). Exactly one winner,
                                 also when the old epoch died and every survivor races for the next one.
    <run>/epochs/e000042.json    the roster: members in rank order, rank 0's ip + port (the TCPStore of the epoch),
                                 src_rank (whose state everyone receives), committed step, reason.
    <run>/leave/<job id>         a file a job's batch shell touches on SIGTERM (or anyone, to send a job away);
                                 leave/ALL sends every job away (they checkpoint first).
    <run>/STOP                   stop the run at the next step (final checkpoint, DONE).
    <run>/events.jsonl           epoch / failure / join / leave / checkpoint records, one JSON line each.

Only plain files with atomic create / replace (same rules as perf_kit.pool.store), so it works on Lustre and NTFS.
The collective group itself is NOT in the files: each epoch is a fresh torch.distributed group (NCCL over TCP on the
cluster, gloo in the CPU tests) whose TCPStore runs on the epoch's rank 0.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from ..pool.store import read_json, write_json

_EP = re.compile(r"^e(\d{6})\.json$")
_CL = re.compile(r"^e(\d{6})\.claim$")


def iface_ip(prefix: str = "hsn0") -> str:
    """IPv4 of the first interface whose name starts with `prefix` (Linux `ip -4 -br addr`); 127.0.0.1 otherwise."""
    if prefix in ("", "lo", "127.0.0.1"):
        return "127.0.0.1"
    try:
        out = subprocess.run(["ip", "-4", "-br", "addr"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            parts = line.split()
            if parts and parts[0].startswith(prefix) and len(parts) >= 3:
                return parts[2].split("/")[0]
    except Exception:                                       # noqa: BLE001  (no `ip` on Windows)
        pass
    return "127.0.0.1"


def free_port(ip: str) -> int:
    s = socket.socket()
    s.bind((ip, 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Membership:
    def __init__(self, run_dir, mid: str, info: dict, hb_s: float = 5.0):
        self.root = Path(run_dir)
        for d in ("members", "epochs", "leave", "ckpt", "metrics"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.mid = mid
        self.info = dict(info)          # ip, job, local_rank, job_size, host, pid, joined
        self.fields = {"state": "joining", "epoch": -1, "step": 0}
        self.hb_s = hb_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.beat()

    # ---------------------------------------------------------------- own file
    def update(self, **kw):
        with self._lock:
            self.fields.update(kw)
        self.beat()

    def beat(self):
        with self._lock:
            rec = {**self.info, **self.fields, "mid": self.mid, "beat": time.time()}
        write_json(self.root / "members" / f"{self.mid}.json", rec)

    def start_heartbeat(self):
        def loop():
            while not self._stop.wait(self.hb_s):
                try:
                    self.beat()
                except Exception:                           # noqa: BLE001  (a missed beat is not fatal)
                    pass
        self._thread = threading.Thread(target=loop, name="elastic-heartbeat", daemon=True)
        self._thread.start()

    def leave(self):
        self._stop.set()
        try:
            os.unlink(self.root / "members" / f"{self.mid}.json")
        except OSError:
            pass

    # ---------------------------------------------------------------- others
    def members(self, fresh_s: float | None = None) -> list[dict]:
        now = time.time()
        out = []
        for p in (self.root / "members").glob("*.json"):
            d = read_json(p)
            if d and (fresh_s is None or now - d.get("beat", 0) <= fresh_s):
                out.append(d)
        return out

    def complete_joiners(self, fresh_s: float, settle_s: float, exclude=(), job_wait_s: float = 60.0) -> list[dict]:
        """Fresh `joining` members whose whole job is present (all `job_size` processes), or whose job's first
        process has waited `job_wait_s`; each waited at least `settle_s`. Ordered by job arrival, then local rank."""
        now = time.time()
        jobs = {}
        for m in self.members(fresh_s):
            if m.get("state") != "joining" or m["mid"] in exclude:
                continue
            jobs.setdefault(m["job"], []).append(m)
        out = []
        for job, ms in sorted(jobs.items(), key=lambda kv: min(m["joined"] for m in kv[1])):
            first = min(m["joined"] for m in ms)
            if now - max(m.get("ready", m["joined"]) for m in ms) < settle_s:
                continue
            if len(ms) < int(ms[0].get("job_size", 1)) and now - first < job_wait_s:
                continue
            out.extend(sorted(ms, key=lambda m: m.get("local_rank", 0)))
        return out

    # ---------------------------------------------------------------- epochs
    def _epath(self, n: int) -> Path:
        return self.root / "epochs" / f"e{n:06d}.json"

    def latest_epoch(self) -> int:
        """Highest epoch number with a roster or a claim (-1: none)."""
        best = -1
        for p in (self.root / "epochs").iterdir():
            m = _EP.match(p.name) or _CL.match(p.name)
            if m:
                best = max(best, int(m.group(1)))
        return best

    def latest_roster(self) -> tuple[int, dict | None]:
        best, rec = -1, None
        for p in (self.root / "epochs").iterdir():
            m = _EP.match(p.name)
            if m and int(m.group(1)) > best:
                best = int(m.group(1))
        if best >= 0:
            rec = read_json(self._epath(best))
        return best, rec

    def roster(self, n: int) -> dict | None:
        return read_json(self._epath(n))

    def claim(self, n: int) -> bool:
        p = self.root / "epochs" / f"e{n:06d}.claim"
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(fd, json.dumps({"mid": self.mid, "t": time.time()}).encode())
        os.close(fd)
        return True

    def claim_age(self, n: int) -> float | None:
        try:
            return time.time() - (self.root / "epochs" / f"e{n:06d}.claim").stat().st_mtime
        except FileNotFoundError:
            return None

    def write_roster(self, n: int, rec: dict) -> None:
        p = self._epath(n)
        tmp = p.with_name(f".{p.name}.{self.mid}.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, p)

    # ---------------------------------------------------------------- flags + log
    def leave_requested(self, job: str) -> bool:
        d = self.root / "leave"
        extra = os.environ.get("ELASTIC_LEAVE_FILE")       # the job script's own flag (its TERM trap touches it)
        return (d / str(job)).exists() or (d / "ALL").exists() or bool(extra and os.path.exists(extra))

    def stop_requested(self) -> bool:
        return (self.root / "STOP").exists()

    def done(self) -> bool:
        return (self.root / "DONE").exists()

    def event(self, rec: dict) -> None:
        rec = {"t": round(time.time(), 3), **rec}
        with open(self.root / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def events(self) -> list[dict]:
        p = self.root / "events.jsonl"
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
