"""Submit N small worker jobs for one ElasticTCP run (login node, no torch needed).

    export SITE_ENV=... REPO=... EL_SCRIPT=perf_kit/elastic/example.py EL_ARGS="--mode elastic --out RUN ..."
    python -m perf_kit.elastic.submit --run RUN --n 8 --gpus 1 --time 02:00:00 --budget_gpu_hours 20 \\
        [--stagger 0] [--cpus_per_gpu 16] [--mem_per_gpu 60G] [--name el] [--dry_run] -- -A <acct> -p <part>

Every job gets the caller's environment (sbatch inherits it; EL_RUN is set to --run): long argument strings never go
through `--export=...,ARGS="..."`, which splits at commas. Jobs are 1 node each (elastic jobs never span nodes: the
group across jobs is built by the run itself). Refuses when GPUs x time limit x jobs would exceed --budget_gpu_hours
(counting every job already in RUN/launch/jobs.jsonl), or when one call asks for more than --max_gpus. Submitted ids go to RUN/launch/jobs.jsonl.
Use the same --run later to add jobs to a running run (it joins them at the next step boundary).
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

from ..pool.launch import _hours, _ledger


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, required=True, help="jobs to submit now")
    ap.add_argument("--gpus", type=int, default=1, help="GPUs per job (1-4, one node)")
    ap.add_argument("--time", default="01:00:00")
    ap.add_argument("--budget_gpu_hours", type=float, required=True)
    ap.add_argument("--max_gpus", type=int, default=16, help="cap on GPUs of all jobs in the ledger")
    ap.add_argument("--stagger", type=float, default=0.0, help="seconds between the jobs' earliest start times")
    ap.add_argument("--cpus_per_gpu", type=int, default=16)
    ap.add_argument("--mem_per_gpu", default="60G")
    ap.add_argument("--name", default="elastic")
    ap.add_argument("--sbatch", default=str(Path(__file__).resolve().parents[2] / "bench" / "elastic" /
                                           "elastic_job.sbatch"))
    ap.add_argument("--sbatch_bin", default="sbatch")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra sbatch arguments (after --)")
    a = ap.parse_args(argv)
    if not 1 <= a.gpus <= 4:
        raise SystemExit("an elastic job is one node: 1-4 GPUs")
    run = Path(a.run)
    (run / "launch").mkdir(parents=True, exist_ok=True)
    jobs = _ledger(run)
    booked = sum(j["gpu_hours"] for j in jobs)
    gh = a.gpus * _hours(a.time)
    if booked + a.n * gh > a.budget_gpu_hours + 1e-9:
        raise SystemExit(f"refusing: {booked:.2f} GPU-h booked + {a.n} x {gh:.2f} > budget {a.budget_gpu_hours}")
    if a.n * a.gpus > a.max_gpus:
        raise SystemExit(f"refusing: {a.n} x {a.gpus} GPUs in one call > max_gpus {a.max_gpus}")
    mem = a.mem_per_gpu
    m = re.fullmatch(r"(\d+)([GM])", mem)
    mem_job = f"{int(m.group(1)) * a.gpus}{m.group(2)}" if m else mem
    env = {**os.environ, "EL_RUN": str(run)}
    out = []
    for i in range(a.n):
        cmd = [*shlex.split(a.sbatch_bin), "--parsable", "-J", a.name, "--nodes", "1", "--gpus-per-node",
               str(a.gpus), "--ntasks-per-node", str(a.gpus), "--cpus-per-task", str(a.cpus_per_gpu), "--mem",
               mem_job, "-t", a.time]
        if a.stagger and i:
            cmd += ["--begin", f"now+{int(a.stagger * i)}"]
        cmd += [*a.extra, a.sbatch]
        if a.dry_run:
            print("[dry-run]", " ".join(shlex.quote(c) for c in cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"sbatch failed: {r.stderr.strip()[:400]}")
        jid = r.stdout.strip().split(";")[0]
        rec = {"id": jid, "gpus": a.gpus, "nodes": 1, "time": a.time, "gpu_hours": gh, "t": time.time()}
        with open(run / "launch" / "jobs.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        out.append(jid)
    print(json.dumps({"submitted": out, "booked_gpu_hours": round(booked + len(out) * gh, 3)}))


if __name__ == "__main__":
    main()
