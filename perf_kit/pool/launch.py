"""Keep a pool topped up with small Slurm jobs, under a GPU cap and a hard GPU-hour budget.

    python -m perf_kit.pool.launch --run RUN_DIR --sbatch bench/pool/pool_job.sbatch --gpus 1 --time 00:30:00 \\
        --max_gpus 16 --max_pending 2 --max_jobs 40 --budget_gpu_hours 20 [--loop 300] [--dry_run] -- <sbatch args>

One call = one top-up: count this run's own jobs (by the job ids in RUN_DIR/launch/jobs.jsonl, never by user or
name), and submit more while ALL of these hold:
  * the run is not DONE;
  * GPUs of its pending + running jobs + the new job <= --max_gpus (so queued jobs count against the cap too);
  * pending jobs < --max_pending (keep a couple in the queue: small jobs start within minutes on a busy cluster);
  * jobs ever submitted < --max_jobs, and the GPU-hours *booked* (GPUs x time limit, summed over every job ever
    submitted, an upper bound on what they can use) + the new job <= --budget_gpu_hours.
--loop S repeats every S seconds until one of the stops is hit. A worker job can also call one top-up as its last
line (`POOL_TOPUP=1` in bench/pool/pool_job.sbatch) so the pool keeps itself alive without a login-node loop.
Nodes per job are capped at 4. Every job gets the run's environment (POOL_RUN, ...) by inheritance, never through
`sbatch --export=...,ARGS=...`, which splits values at commas.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from .store import Store, read_json

MAX_NODES = 4


def _hours(limit: str) -> float:
    """Slurm time limit ([D-]HH:MM:SS, MM:SS or MM) -> hours."""
    days = 0
    if "-" in limit:
        d, limit = limit.split("-", 1)
        days = int(d)
    parts = [int(x) for x in limit.split(":")]
    if len(parts) == 1:
        h, m, s = 0, parts[0], 0
    elif len(parts) == 2:
        h, (m, s) = 0, parts
    else:
        h, m, s = parts
    return days * 24 + h + m / 60 + s / 3600


def _ledger(run: Path) -> list[dict]:
    p = run / "launch" / "jobs.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _states(ids: list[str], squeue: str) -> dict[str, str]:
    if not ids:
        return {}
    def ask(chunk):
        r = subprocess.run([*shlex.split(squeue), "-h", "-j", ",".join(chunk), "-o", "%i %T"], capture_output=True,
                           text=True)
        return r.returncode, r.stdout
    rc, out = ask(ids)
    if rc != 0:                       # some Slurm versions refuse the whole list if one id is long purged
        out = "".join(ask([i])[1] for i in ids)
    st = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            st[parts[0]] = parts[1]
    return st     # ids absent from squeue have finished (or never existed)


def topup(a) -> dict:
    run = Path(a.run)
    (run / "launch").mkdir(parents=True, exist_ok=True)
    store = Store(run)
    gpus_job = a.gpus * a.nodes
    if a.nodes > MAX_NODES:
        raise SystemExit(f"refusing: {a.nodes} nodes per job > cap of {MAX_NODES}")
    submitted = []
    with store.lock(wait=60):
        jobs = _ledger(run)
        states = _states([j["id"] for j in jobs if j["id"] != "dry"], a.squeue)
        active = [j for j in jobs if states.get(j["id"]) in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING")]
        pending = [j for j in active if states.get(j["id"]) == "PENDING"]
        used = sum(j["gpus"] for j in active)
        booked = sum(j["gpu_hours"] for j in jobs)
        n_jobs = len(jobs)
        why = None
        while True:
            if store.done():
                why = "run is DONE"
                break
            if used + gpus_job > a.max_gpus:
                why = f"cap: {used} GPUs queued/running + {gpus_job} > {a.max_gpus}"
                break
            if len(pending) >= a.max_pending:
                why = f"{len(pending)} pending >= max_pending {a.max_pending}"
                break
            if n_jobs >= a.max_jobs:
                why = f"max_jobs {a.max_jobs} reached"
                break
            gh = gpus_job * _hours(a.time)
            if booked + gh > a.budget_gpu_hours + 1e-9:
                why = f"budget: {booked:.2f} + {gh:.2f} GPU-h > {a.budget_gpu_hours}"
                break
            cmd = [*shlex.split(a.sbatch_bin), "--parsable", "-J", a.name, "--nodes", str(a.nodes), "--gpus-per-node", str(a.gpus),
                   "--ntasks-per-node", str(a.gpus), "-t", a.time, *a.extra, a.sbatch]
            env = {**os.environ, "POOL_RUN": str(run)}
            if a.dry_run:
                jid = "dry"
                print("[dry-run]", " ".join(cmd))
            else:
                r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                if r.returncode != 0:
                    why = f"sbatch failed: {r.stderr.strip()[:300]}"
                    break
                jid = r.stdout.strip().split(";")[0]
                if not re.fullmatch(r"\d+", jid):
                    why = f"unexpected sbatch output {r.stdout!r}"
                    break
            rec = {"id": jid, "gpus": gpus_job, "nodes": a.nodes, "time": a.time, "gpu_hours": gh, "t": time.time()}
            if not a.dry_run:
                with open(run / "launch" / "jobs.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            submitted.append(rec)
            used += gpus_job
            booked += gh
            n_jobs += 1
            pending.append(rec)
            if a.dry_run:
                why = "dry run: one submission shown"
                break
    res = {"submitted": [j["id"] for j in submitted], "active_gpus": used, "booked_gpu_hours": round(booked, 3),
           "jobs_ever": n_jobs, "stop": why, "done": store.done()}
    print(json.dumps(res))
    return res


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sbatch", required=True, help="the job script (bench/pool/pool_job.sbatch or your own)")
    ap.add_argument("--name", default="pool", help="job name (for humans; jobs are counted by id)")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs per node per job")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--time", default="00:30:00")
    ap.add_argument("--max_gpus", type=int, default=16)
    ap.add_argument("--max_pending", type=int, default=2)
    ap.add_argument("--max_jobs", type=int, default=20)
    ap.add_argument("--budget_gpu_hours", type=float, required=True)
    ap.add_argument("--loop", type=float, default=0.0, help="repeat every S seconds until a hard stop")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--sbatch_bin", default="sbatch")
    ap.add_argument("--squeue", default="squeue")
    ap.add_argument("extra", nargs="*", help="extra sbatch arguments (after --)")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    while True:
        r = topup(a)
        hard = r["done"] or (r["stop"] or "").startswith(("max_jobs", "budget", "sbatch failed", "unexpected"))
        if not a.loop or hard or a.dry_run:
            return
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
