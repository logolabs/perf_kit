"""Pilot gate: does a short run earn a long one? Reads a trainer's log.jsonl (one JSON record per line: step lines with
`step`, `step_s`, `mfu`; `val` events with `val_*_loss`; an optional `profile` event) and checks it against targets:

  * steady-state MFU (median over the step lines after `warm` steps) >= `mfu_min`;
  * every `val_*_loss` at the reference's steps within `val_tol` (relative) of the reference run's;
  * GPU idle share from the profile (if logged) <= `idle_max`.

    python -m perf_kit.gate log.jsonl --mfu_min 0.30 [--ref ref_log.jsonl --val_tol 0.005] [--idle_max 0.15]

Exit 0 = pass, 1 = fail; the verdict is printed as one JSON line either way.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def steady(recs, key: str, warm: int):
    xs = [r[key] for r in recs if isinstance(r.get("step"), int) and "event" not in r and r.get(key) is not None
          and r["step"] > warm]
    return statistics.median(xs) if xs else None


def vals(recs) -> dict:
    v = {}
    for r in recs:
        if r.get("event") == "val":
            for k, x in r.items():
                if k.startswith("val_") and x is not None:
                    v[(r["step"], k)] = x
    return v


def verdict(recs, mfu_min: float, ref=None, val_tol: float = 0.005, idle_max=None, warm: int = 20) -> dict:
    out = {"mfu": steady(recs, "mfu", warm), "step_s": steady(recs, "step_s", warm), "checks": {}}
    out["checks"]["mfu"] = out["mfu"] is not None and out["mfu"] >= mfu_min
    if ref is not None:
        mine, theirs = vals(recs), vals(ref)
        common = sorted(set(mine) & set(theirs))
        rel = {f"{s}:{k}": round((mine[(s, k)] - theirs[(s, k)]) / theirs[(s, k)], 5) for s, k in common}
        out["val_rel"] = rel
        out["checks"]["val"] = bool(common) and all(abs(x) <= val_tol for x in rel.values())
        rs = steady(ref, "step_s", warm)
        if rs and out["step_s"]:
            out["speedup_vs_ref"] = round(rs / out["step_s"], 3)
    prof = [r for r in recs if r.get("event") == "profile"]
    if prof:
        out["gpu_idle"] = prof[-1].get("gpu_idle")
        out["kernels_per_step"] = prof[-1].get("kernels_per_step")
        if idle_max is not None and out["gpu_idle"] is not None:
            out["checks"]["idle"] = out["gpu_idle"] <= idle_max
    out["pass"] = all(out["checks"].values())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--mfu_min", type=float, required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--val_tol", type=float, default=0.005)
    ap.add_argument("--idle_max", type=float, default=None)
    ap.add_argument("--warm", type=int, default=20)
    a = ap.parse_args(argv)
    v = verdict(load(a.log), a.mfu_min, load(a.ref) if a.ref else None, a.val_tol, a.idle_max, a.warm)
    print(json.dumps(v))
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
