"""Host <-> GPU copy bandwidth on a GH200 node (NVLink-C2C): pinned 1 GB buffers, H2D and D2H, each GPU alone and all
visible GPUs at once (one thread + stream per GPU). Decides whether CPU activation offload can hide behind compute
(offload traffic per step / bandwidth vs the step time) instead of recomputation.

Usage: python gh200_node.py --out result.json [--gb 1] [--iters 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch


def one(dev: int, gb: float, iters: int, res: dict, key: str, barrier=None):
    torch.cuda.set_device(dev)
    n = int(gb * 2**30)
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dev}")
    s = torch.cuda.Stream(dev)
    with torch.cuda.stream(s):
        d.copy_(h, non_blocking=True)
        h.copy_(d, non_blocking=True)
    s.synchronize()
    out = {}
    for name, (dst, src) in {"h2d": (d, h), "d2h": (h, d)}.items():
        if barrier is not None:
            barrier.wait()
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            for _ in range(iters):
                dst.copy_(src, non_blocking=True)
        s.synchronize()
        out[name + "_gbps"] = round(n * iters / (time.perf_counter() - t0) / 1e9, 1)
    res[key] = out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gb", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args(argv)
    ng = torch.cuda.device_count()
    res = {"gpus": ng, "names": [torch.cuda.get_device_name(i) for i in range(ng)]}
    for i in range(ng):
        one(i, a.gb, a.iters, res, f"alone_gpu{i}")
    if ng > 1:
        b = threading.Barrier(ng)
        th = [threading.Thread(target=one, args=(i, a.gb, a.iters, res, f"together_gpu{i}", b)) for i in range(ng)]
        [t.start() for t in th]
        [t.join() for t in th]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
