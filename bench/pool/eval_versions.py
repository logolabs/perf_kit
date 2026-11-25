"""Evaluate kept global versions of an example_lm pool run on the validation set (bits per byte).

    python bench/pool/eval_versions.py --run RUN_DIR --val val.jsonl [--versions 24 25 26] [--no_lookahead]

--no_lookahead also evaluates each version with the last Nesterov look-ahead removed (theta + lr * momentum * m),
to tell a bad point of the outer trajectory from extrapolation by the outer momentum.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from perf_kit.pool.example_lm import GPT, evaluate, load_bytes
from perf_kit.pool.store import Store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--versions", type=int, nargs="*")
    ap.add_argument("--no_lookahead", action="store_true")
    ap.add_argument("--val_windows", type=int, default=2048)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = torch.from_numpy(load_bytes(a.val).copy()).to(dev)
    for run in a.run:
        run_one(run, a, val, dev)


def run_one(run, a, val, dev):
    meta = json.load(open(sorted(glob.glob(os.path.join(run, "metrics", "*.json")))[0]))["args"]
    cfg = json.load(open(os.path.join(run, "config.json")))
    model = GPT(meta["ctx"], meta["dim"], meta["layers"], meta["heads"]).to(dev)
    st = Store(run)
    for v in a.versions or st.versions():
        s = st.load(v, full=a.no_lookahead)
        arms = [("as_saved", s.params)]
        if a.no_lookahead and s.m is not None:
            k = cfg["outer"]["lr"] * cfg["outer"]["momentum"]
            arms.append(("no_lookahead", {n: p + k * s.m[n] for n, p in s.params.items()}))
        for tag, params in arms:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(params[n].to(p.dtype))
            bpb = evaluate(model, val, meta["ctx"], a.val_windows, dev)
            rec = {"run": os.path.basename(run), "version": v, "arm": tag, "samples": s.samples, "outer_steps": s.outer_steps, "val_bpb": bpb}
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
