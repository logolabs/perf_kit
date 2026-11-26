"""Worked example: a byte-level GPT trained either with plain synchronous DDP or on the pool, same code otherwise.

    # synchronous DDP (srun / torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT)
    python -m perf_kit.pool.example_lm --mode ddp  --out RUN --train train.jsonl --val val.jsonl --total_samples N
    # one pool worker (any number of GPUs, DDP inside); start as many as you like, whenever
    python -m perf_kit.pool.example_lm --mode pool --out RUN --train train.jsonl --val val.jsonl --total_samples N

Data: JSONL rows with "messages" (contents joined) or "prompt"/"completion"; UTF-8 bytes are the tokens (vocab 256),
random windows of --ctx bytes. The validation loss is the mean over a fixed set of windows of val.jsonl, in bits per
byte. --total_samples counts training windows over the whole run (all workers). The learning-rate schedule (linear
warm-up, cosine to --min_lr_frac) follows samples done: DDP's own count, or the pool's progress (--schedule_by:
`progress_est`, the estimate that counts the whole pool's live round, by default; `progress`, applied samples plus
this job's own, is what the 2026-11-24 comparison in WRITEUP section 10 ran with, and it under-counts until a job's
first outer sync). --local_warmup N also ramps each pool job's lr over its own first N steps (fresh inner optimizer).
Loop: no host syncs in the step (the stop flag is agreed every --log_every steps), fused AdamW, bf16 autocast, DDP
gradients as bucket views, optional torch.compile (--compile).
Per job, RUN/metrics/<job>.json records times (process start, first step, time in steps, in pool syncs, in eval)
so the wall time, GPU-hours and idle time of a run can be added up afterwards (bench/pool/summarize.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_PROC = time.time()


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, ctx=256, d=384, layers=6, heads=6):
        super().__init__()
        self.tok, self.pos = nn.Embedding(256, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, tgt):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.ln(x) @ self.tok.weight.T            # tied output embedding
        return F.cross_entropy(logits.float().view(-1, 256), tgt.reshape(-1))


# ---------------------------------------------------------------- data
def load_bytes(path) -> np.ndarray:
    parts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "messages" in r:
                t = "\n".join(m.get("content", "") for m in r["messages"])
            else:
                t = r.get("prompt", "") + r.get("completion", "")
            parts.append(t.encode("utf-8") + b"\n\n")
    return np.frombuffer(b"".join(parts), dtype=np.uint8)


def windows(data: torch.Tensor, offsets: torch.Tensor, ctx: int):
    idx = offsets[:, None] + torch.arange(ctx + 1, device=data.device)[None]
    w = data[idx].long()
    return w[:, :-1], w[:, 1:]


def batch(data, n, ctx, seed, device):
    g = torch.Generator().manual_seed(int(seed) & ((1 << 63) - 1))
    off = torch.randint(0, data.numel() - ctx - 1, (n,), generator=g).to(device)
    return windows(data, off, ctx)


@torch.no_grad()
def evaluate(model, val, ctx, n_windows, device, bs=128):
    g = torch.Generator().manual_seed(0)
    off = torch.randint(0, val.numel() - ctx - 1, (n_windows,), generator=g).to(device)
    m = model.module if hasattr(model, "module") else model
    m.eval()
    tot = 0.0
    for i in range(0, n_windows, bs):
        x, y = windows(val, off[i:i + bs], ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tot += float(m(x, y)) * x.shape[0]
    m.train()
    return tot / n_windows / math.log(2)                   # bits per byte


def lr_at(progress, a):
    if progress < a.warmup_frac:
        return a.lr * progress / a.warmup_frac
    p = (progress - a.warmup_frac) / max(1e-9, 1 - a.warmup_frac)
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ddp", "pool"], required=True)
    ap.add_argument("--out", required=True, help="DDP: output dir; pool: the pool's run folder")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--total_samples", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64, help="windows per GPU per step")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--val_windows", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner_steps", type=int, default=500)
    ap.add_argument("--outer_lr", type=float, default=0.7)
    ap.add_argument("--outer_momentum", type=float, default=0.9)
    ap.add_argument("--outer_mode", default="dn", choices=["dn", "buffered"])
    ap.add_argument("--max_gpus", type=int, default=16)
    ap.add_argument("--keep_every", type=int, default=0, help="pool: keep every Nth global version (curves)")
    ap.add_argument("--schedule_by", default="progress_est", choices=["progress_est", "progress"],
                    help="pool: which progress drives the lr schedule (progress = the 2026-11-24 comparison)")
    ap.add_argument("--local_warmup", type=int, default=0, help="pool: also ramp lr over this job's first N steps")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--die_after_s", type=float, default=0.0, help="test hook: hard-kill every rank after S s of "
                    "training (os._exit, no cleanup, mid-round), to simulate a crashed job")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--cpu", action="store_true", help="force CPU (gloo) even if CUDA is visible")
    a = ap.parse_args(argv)

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", device_id=device
                                if device.type == "cuda" else None)
    torch.backends.cuda.matmul.allow_tf32 = True
    job = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    train = torch.from_numpy(load_bytes(a.train).copy()).to(device)
    val = torch.from_numpy(load_bytes(a.val).copy()).to(device)
    torch.manual_seed(a.seed)                               # same init on every rank and every worker
    model = GPT(a.ctx, a.dim, a.layers, a.heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    fwd = torch.compile(model) if a.compile else model      # the pool unwraps DDP / compile to the raw names
    ddp = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[local] if device.type == "cuda" else None,
                                                    gradient_as_bucket_view=True) if world > 1 else fwd
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)

    pool = None
    if a.mode == "pool":
        from perf_kit.pool import Pool
        pool = Pool(a.out, model, inner_steps=a.inner_steps, total_samples=a.total_samples, gpus=world,
                    max_gpus=a.max_gpus, outer_lr=a.outer_lr, outer_momentum=a.outer_momentum, mode=a.outer_mode,
                    seed=a.seed, keep_every=a.keep_every, step_samples=a.batch * world)
        if pool.full or pool.done:
            if rank == 0:
                print(f"[pool] {'full' if pool.full else 'done'}: this job leaves without training")
            return

    step_samples = a.batch * world
    rec = {"job": job, "mode": a.mode, "gpus": world, "params": n_params, "t_proc": T_PROC, "t_ready": time.time(),
           "t_first_step": None, "train_s": 0.0, "sync_s": 0.0, "eval_s": 0.0, "steps": 0, "samples": 0,
           "curve": [], "args": vars(a)}

    def save(final=False):
        if rank != 0:
            return
        rec["t_end"] = time.time()
        if pool is not None:
            rec["pool"] = pool.stats()
        rec["final"] = final
        path = os.path.join(a.out, "metrics", f"{job}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(path + ".tmp", path)

    samples_done = 0
    lr_sum = 0.0                                            # sum of the lrs used (ranks of a job must agree)
    loss_ema = None
    t_train0 = None
    last_save = time.time()
    step = 0
    while True:
        progress = getattr(pool, a.schedule_by) if pool else samples_done / a.total_samples
        if (pool and pool.done) or (not pool and samples_done >= a.total_samples):
            break
        if world > 1 and step % a.log_every == 0:          # every rank must agree to stop at the same step; checked
            f = torch.tensor([float(stop["flag"])], device=device)   # at log steps only (a host sync)
            torch.distributed.all_reduce(f)
            stop["flag"] = bool(f.item())
        if stop["flag"] and (world == 1 or step % a.log_every == 0):
            break
        lr = lr_at(progress, a)
        if pool and a.local_warmup:                         # a joining job's inner AdamW starts with empty moments
            lr *= min(1.0, (pool.local_steps + 1) / a.local_warmup)
        lr_sum += lr
        for g in opt.param_groups:
            g["lr"] = lr
        seed = (pool.seed() + 7919 * (pool._steps + 1) if pool else a.seed * 1_000_003 + step) * 64 + rank
        x, y = batch(train, a.batch, a.ctx, seed, device)
        t0 = time.time()
        if rec["t_first_step"] is None:
            rec["t_first_step"] = t0
            t_train0 = t0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = ddp(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        samples_done += step_samples
        if step % a.log_every == 0:
            lv = float(loss) / math.log(2)
            loss_ema = lv if loss_ema is None else 0.9 * loss_ema + 0.1 * lv
            rec["curve"].append([round(time.time() - T_PROC, 2), round(progress, 5), round(lv, 4)])
            if device.type == "cuda":
                torch.cuda.synchronize()
        rec["train_s"] = time.time() - t_train0          # wall time since the first step (steps + syncs + evals)
        rec["steps"], rec["samples"] = step, samples_done
        if pool:
            ts = time.time()
            if pool.sync(ddp, step_samples) and rank == 0:
                print(f"[pool {job}] v{pool.version} applied {pool.applied_samples}/{a.total_samples} "
                      f"loss {loss_ema if loss_ema is not None else float('nan'):.3f} bpb", flush=True)
            rec["sync_s"] += time.time() - ts
        elif rank == 0 and step % (a.log_every * 10) == 0:
            print(f"[ddp {job}] step {step} samples {samples_done}/{a.total_samples} loss {loss_ema:.3f} bpb",
                  flush=True)
        if a.die_after_s and time.time() - t_train0 > a.die_after_s:
            save()
            os._exit(9)                                     # simulated crash: no close(), no cleanup
        if time.time() - last_save > 30:
            save()
            last_save = time.time()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end_train = time.time()
    if pool:
        pool.close(ddp)
        if rank == 0 and pool.done:                         # the global result, evaluated by whoever finishes
            pool.load_global(model)
    te = time.time()
    if rank == 0 and (not pool or pool.done):
        rec["val_bpb"] = evaluate(model, val, a.ctx, a.val_windows, device)
        if pool:                                            # also the stored (look-ahead) weights, for the record
            rec["val_version"] = pool.load_global(model, lookahead=True)
            rec["val_bpb_lookahead"] = evaluate(model, val, a.ctx, a.val_windows, device)
    rec["eval_s"] = time.time() - te
    rec["train_s"] = t_end_train - (t_train0 or t_end_train)
    save(final=True)
    print(json.dumps({**{k: rec.get(k) for k in ("job", "mode", "gpus", "steps", "samples", "train_s", "sync_s",
                                                 "val_bpb")}, "rank": rank, "lr_sum": lr_sum}), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
