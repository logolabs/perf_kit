"""A fake pool worker for the CPU tests: float64 linear regression through the real Pool API and run folder.

    python -m perf_kit.pool.sim --run DIR --worker w0 [--gpus 1] [--start_delay 0] [--die_after_rounds N]
                                [--die_in inner|lock] [--slow 0.0]

Prints one JSON line at the end (what it did). --die_after_rounds kills the process with os._exit (no cleanup, no
leave), either mid-round (`inner`) or while holding the run lock (`lock`), to test dead workers and lock takeover.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

D = 16


def target():
    g = torch.Generator().manual_seed(1234)
    return torch.randn(D, 1, generator=g, dtype=torch.float64), torch.randn(1, generator=g, dtype=torch.float64)


def batch(seed: int, n: int):
    g = torch.Generator().manual_seed(seed)
    w, b = target()
    x = torch.randn(n, D, generator=g, dtype=torch.float64)
    y = x @ w + b + 0.01 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    return x, y


def model():
    torch.manual_seed(0)                                  # every worker builds the same initial model
    return torch.nn.Linear(D, 1).double()


def excess_loss(m) -> float:
    w, b = target()
    return float(((m.weight.detach().T - w) ** 2).sum() + ((m.bias.detach() - b) ** 2).sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--worker", required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--max_gpus", type=int, default=16)
    ap.add_argument("--inner_steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8, help="samples per step per 'GPU'")
    ap.add_argument("--total_samples", type=int, default=6000)
    ap.add_argument("--start_delay", type=float, default=0.0)
    ap.add_argument("--die_after_rounds", type=int, default=-1)
    ap.add_argument("--die_in", default="inner", choices=["inner", "lock"])
    ap.add_argument("--slow", type=float, default=0.0, help="seconds per inner step")
    ap.add_argument("--lock_timeout", type=float, default=3.0)
    a = ap.parse_args(argv)
    from perf_kit.pool import Pool
    import perf_kit
    time.sleep(a.start_delay)
    m = model()
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    pool = Pool(a.run, m, inner_steps=a.inner_steps, total_samples=a.total_samples, gpus=a.gpus,
                max_gpus=a.max_gpus, outer_lr=0.7, outer_momentum=0.9, worker_id=a.worker, heartbeat_s=0.3,
                heartbeat_timeout=2.0, lock_timeout=a.lock_timeout, state_dtype=torch.float64, keep_versions=4,
                log=lambda *x: None)
    out = {"worker": a.worker, "full": pool.full, "index": pool.index, "perf_kit": os.path.dirname(perf_kit.__file__)}
    if pool.full or pool.done:
        print(json.dumps({**out, "rounds": 0}))
        return
    n = a.batch * a.gpus
    rounds = 0
    while not pool.done:
        x, y = batch(pool.seed() + pool._steps, n)
        loss = torch.nn.functional.mse_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if a.slow:
            time.sleep(a.slow)
        if a.die_after_rounds >= 0 and rounds >= a.die_after_rounds:
            if a.die_in == "inner" and pool._steps == a.inner_steps // 2:
                os._exit(17)
            if a.die_in == "lock" and pool._steps == a.inner_steps - 1:
                pool._publish(m)
                with pool.store.lock():
                    os._exit(18)                          # dies holding the lock
        if pool.sync(m, n):
            rounds += 1
    pool.close(m)
    pool.load_global(m)
    print(json.dumps({**out, "rounds": rounds, "contributions": pool._seq, "excess": excess_loss(m),
                      "version": pool.version}))


if __name__ == "__main__":
    sys.exit(main())
