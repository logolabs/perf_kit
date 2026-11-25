"""Add up a DDP run and a pool run: final validation, wall time, GPU-hours, idle time, queue wait.

    python bench/pool/summarize.py --run RUN_DIR [--run RUN_DIR ...] [--sacct]   (on the cluster: --sacct asks
    sacct for Submit / Start / End / Elapsed of every job id found in RUN_DIR/metrics)

Per job (metrics/<job>.json written by perf_kit/pool/example_lm.py):
  busy   = time from its first training step to its last (steps + pool syncs + logging);
  sync   = time inside pool.sync (publish, apply, pull, broadcast);
  idle   = allocated time not in `busy`: start-up (imports, data load, model build, joining) + final eval + exit.
           For a killed job, its unpublished partial round is counted in `lost`, not `busy`.
GPU-hours = GPUs x allocated elapsed (sacct when available, else process lifetime).
Run wall time = first job start to the last job end (training + final eval); queue wait = eligible (submit, or
the --begin time of a deliberately staggered job) -> start.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import subprocess


def _t(s):
    return dt.datetime.fromisoformat(s).timestamp() if s and s not in ("Unknown", "None") else None


def sacct(ids):
    if not ids:
        return {}
    r = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "-o", "JobID,State,Submit,Eligible,Start,End,ElapsedRaw,AllocTRES"],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        jid, state, sub, elig, start, end, el, tres = line.split("|")
        gpus = 0
        for kv in tres.split(","):
            if kv.startswith("gres/gpu="):
                gpus = int(kv.split("=")[1])
        out[jid] = {"state": state, "submit": _t(sub), "eligible": _t(elig), "start": _t(start), "end": _t(end), "elapsed": float(el or 0),
                    "gpus": gpus}
    return out


def summarize(run, use_sacct):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run, "metrics", "*.json")))]
    sa = sacct([r["job"] for r in recs if r["job"].isdigit()]) if use_sacct else {}
    rows = []
    for r in recs:
        s = sa.get(r["job"], {})
        alloc = s.get("elapsed") or (r["t_end"] - r["t_proc"])
        busy = r["train_s"]
        rows.append({"job": r["job"], "gpus": r["gpus"], "state": s.get("state", "?"), "final": r.get("final"),
                     # queue wait from the moment the job was allowed to start (--begin) to its start
                     "queue_s": (s["start"] - (s.get("eligible") or s["submit"])) if s.get("start") else None,
                     "alloc_s": alloc, "busy_s": busy, "sync_s": r["sync_s"], "idle_s": max(0.0, alloc - busy),
                     "samples": r["samples"], "val_bpb": r.get("val_bpb"), "start": s.get("start") or r["t_proc"],
                     "end": s.get("end") or r["t_end"],
                     "contributions": (r.get("pool") or {}).get("contributions")})
    gh = sum(x["gpus"] * x["alloc_s"] for x in rows) / 3600
    wall = max(x["end"] for x in rows) - min(x["start"] for x in rows) if rows else 0
    done = json.load(open(os.path.join(run, "DONE"))) if os.path.exists(os.path.join(run, "DONE")) else {}
    final = [r for r in recs if r.get("val_bpb") is not None and (not done or r.get("val_version") == done["version"])]
    vals = [r["val_bpb"] for r in final]
    ev = os.path.join(run, "events.jsonl")
    pool = {}
    if os.path.exists(ev):
        E = [json.loads(x) for x in open(ev)]
        app = [e for e in E if e["event"] == "applied"]
        pool = {"versions": max((e["version"] for e in app), default=0), "applied": len(app),
                "dropped": sum(e["event"] == "dropped" for e in E), "applied_samples": max(
                    (e["applied_samples"] for e in app), default=0),
                "mean_staleness": round(sum(e["staleness"] for e in app) / max(1, len(app)), 3),
                "max_staleness": max((e["staleness"] for e in app), default=0)}
    return {"run": os.path.basename(run.rstrip("/")), "jobs": rows, "gpu_hours": round(gh, 4),
            "wall_s": round(wall, 1), "train_samples": sum(x["samples"] for x in rows),
            "busy_gpu_h": round(sum(x["gpus"] * x["busy_s"] for x in rows) / 3600, 4),
            "idle_gpu_h": round(sum(x["gpus"] * x["idle_s"] for x in rows) / 3600, 4),
            "sync_gpu_h": round(sum(x["gpus"] * x["sync_s"] for x in rows) / 3600, 4),
            "val_bpb": vals[-1] if vals else None, "val_version": done.get("version"), "pool": pool}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--sacct", action="store_true")
    a = ap.parse_args()
    for run in a.run:
        print(json.dumps(summarize(run, a.sacct), indent=1))


if __name__ == "__main__":
    main()
