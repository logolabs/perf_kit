"""Worked example and test vehicle: the byte-level GPT of perf_kit.pool.example_lm, trained with ElasticTCP or with
a plain reference loop that does the SAME math (same micro-batch indices per step, same loss scaling, same lr).

    # elastic worker (start as many as you like, whenever; each one = one process / GPU)
    python -m perf_kit.elastic.example --mode elastic --out RUN --train train.jsonl --val val.jsonl \\
        --total_samples N --batch 16 --global_micro 12
    # reference: one process (world 1, G micro-batches accumulated) or plain DDP under torchrun / srun
    python -m perf_kit.elastic.example --mode ref --out REF ... (same flags)

With --global_micro G the optimizer sees the mean gradient of the same G micro-batches (index = global micro cursor,
batch seed = seed * 1_000_003 + index) whatever the world size, so an elastic run with joins, leaves and crashes
must follow the reference curve up to floating-point reduction order (and bf16 rounding with --bf16_hook 1).
Without --global_micro, each rank does --accum micro-batches per step (global batch = world x accum x batch) and
the lr is scaled by sqrt(global batch / --ref_batch).

Test hooks, counted in THIS process's own steps: --die_at_step N (os._exit in the middle of its N-th step's
backward), --hang_at_step N --hang_s S (sleep before the backward: a hung rank), --leave_at_step N (announce leaving
after N steps), --start_delay S, --step_sleep S.
Writes RUN/metrics/elastic.jsonl (rank 0 of each epoch) or REF/metrics/ref.jsonl, and prints one JSON line at the end.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from perf_kit.pool.example_lm import GPT, batch, evaluate, load_bytes, lr_at


def build(a, device):
    torch.manual_seed(a.seed)                               # same init everywhere (a newcomer is overwritten anyway)
    model = GPT(a.ctx, a.dim, a.layers, a.heads).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=device.type == "cuda")
    return model, opt


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def run_elastic(a, device, train, val):
    from perf_kit.elastic import ElasticTCP
    model, opt = build(a, device)
    fwd = torch.compile(model) if a.compile else model

    def warmup():                                           # compile + first-step costs before announcing
        if not a.compile:
            return
        x, y = batch(train, a.batch, a.ctx, 12345, device)
        with autocast(device):
            fwd(x, y).backward()

    el = ElasticTCP(model, opt, a.out, total_samples=a.total_samples, micro_batch=a.batch,
                    global_micro=a.global_micro or None, accum=a.accum, forward_module=fwd,
                    ref_batch=a.ref_batch or None, bf16_hook=bool(a.bf16_hook), clip_norm=1.0,
                    timeout_s=a.timeout_s, hb_s=a.hb_s, dead_s=a.dead_s, gather_s=a.gather_s,
                    found_wait_s=a.found_wait_s, regroup_min_s=a.regroup_min_s, check_every_s=a.check_every_s,
                    ckpt_every_s=a.ckpt_every_s, log_every=a.log_every, chk_every=a.chk_every,
                    init_from=a.init_from or None, warmup=warmup, iface=a.iface, device=device)
    t_train0 = time.time()
    steps_here = 0
    while not el.should_stop():
        with el.guard():
            mis = el.micro_indices()
            for j, mi in enumerate(mis):
                x, y = batch(train, a.batch, a.ctx, a.seed * 1_000_003 + mi, device)
                with el.accumulate(last=j == len(mis) - 1):
                    with autocast(device):
                        loss = el.model(x, y)
                    if a.hang_at_step and steps_here == a.hang_at_step and j == 0:
                        print(f"[test] hanging {a.hang_s} s at step {el.step_n}", flush=True)
                        time.sleep(a.hang_s)
                        a.hang_at_step = 0
                    if a.die_at_step and steps_here == a.die_at_step and j == len(mis) - 1:
                        print(f"[test] dying at step {el.step_n}", flush=True)
                        os._exit(9)
                    el.backward(loss)
            set_lr(opt, lr_at(el.progress, a) * el.lr_scale)
            if el.step():
                steps_here += 1
            if a.hold_at_step and el.step_n == a.hold_at_step:
                with el.hold("test"):                       # long rank-0 work (an evaluation), others wait
                    if el.rank == 0:
                        print(f"[test] rank 0 busy {a.hold_s} s in a hold at step {el.step_n}", flush=True)
                        time.sleep(a.hold_s)
                a.hold_at_step = 0
            if a.leave_at_step and steps_here >= a.leave_at_step:
                el._want_leave = True
            if a.step_sleep:
                time.sleep(a.step_sleep)
    t_train = time.time() - t_train0
    out = {"mode": "elastic", "mid": el.mid, "rank": el.rank, "epoch": el.epoch, "step": el.step_n,
           "samples": el.samples, "steps_here": steps_here, "finished": el._finished, "left": el._left,
           "gave_up": el._gave_up, "train_s": round(t_train, 2), "timing": el.timing}
    if el.final_rank0:                                      # (a job that arrives after DONE trained nothing)
        from perf_kit.elastic.state import fingerprint
        out["sha256"] = fingerprint({"model": model})
        out["val_bpb"] = evaluate(model, val, a.ctx, a.val_windows, device)
    el.close()
    return out


def run_ref(a, device, train, val):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    model, opt = build(a, device)
    fwd = torch.compile(model) if a.compile else model
    ddp = fwd
    if world > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
        ddp = DDP(fwd, device_ids=[device.index] if device.type == "cuda" else None, gradient_as_bucket_view=True)
        if a.bf16_hook:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as H
            ddp.register_comm_hook(state=None, hook=H.bf16_compress_hook)
    G = a.global_micro or world * a.accum
    q, rem = divmod(G, world)
    start, n = rank * q + min(rank, rem), q + (1 if rank < rem else 0)
    counts = [q + (1 if i < rem else 0) for i in range(world)]
    os.makedirs(os.path.join(a.out, "metrics"), exist_ok=True)
    samples, micro, step = 0, 0, 0
    t0 = time.time()
    t_last, walls = None, []
    rows = []
    while samples < a.total_samples:
        tot = torch.zeros((), device=device)
        for j in range(n):
            mi = micro + start + j
            x, y = batch(train, a.batch, a.ctx, a.seed * 1_000_003 + mi, device)
            ctx = ddp.no_sync() if (world > 1 and j < n - 1) else _null()
            with ctx:
                with autocast(device):
                    loss = ddp(x, y)
                (loss * (world / G)).backward()
            tot = tot + loss.detach().float()
        lr = lr_at(samples / a.total_samples, a)
        if not a.global_micro and a.ref_batch:
            lr *= (G * a.batch / a.ref_batch) ** 0.5
        set_lr(opt, lr)
        if world > 1:
            import torch.distributed as dist
            v = torch.zeros(world, dtype=torch.float64, device=device)
            v[rank] = (tot / n).double()
            dist.all_reduce(v)
            losses = v.tolist()
        else:
            losses = [float(tot / n)]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        samples += G * a.batch
        micro += G
        now = time.time()
        if t_last is not None:
            walls.append(now - t_last)
        t_last = now
        if rank == 0 and step % a.log_every == 0:
            row = {"step": step, "samples": samples, "loss": sum(l * k for l, k in zip(losses, counts)) / G,
                   "world": world, "step_s": round(sum(walls[-a.log_every:]) / max(1, len(walls[-a.log_every:])), 5)}
            rows.append(row)
            with open(os.path.join(a.out, "metrics", "ref.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
    out = {"mode": "ref", "rank": rank, "world": world, "step": step, "samples": samples,
           "train_s": round(time.time() - t0, 2)}
    if rank == 0:
        from perf_kit.elastic.state import fingerprint
        out["sha256"] = fingerprint({"model": model})
        out["val_bpb"] = evaluate(model, val, a.ctx, a.val_windows, device)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    return out


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["elastic", "ref"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--total_samples", type=int, required=True)
    ap.add_argument("--batch", type=int, default=16, help="windows per micro-batch")
    ap.add_argument("--global_micro", type=int, default=12, help="micro-batches per optimizer step (0: per-GPU)")
    ap.add_argument("--accum", type=int, default=1, help="per-GPU mode: micro-batches per rank per step")
    ap.add_argument("--ref_batch", type=int, default=0, help="per-GPU mode: lr reference global batch")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--val_windows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16_hook", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--timeout_s", type=float, default=120.0)
    ap.add_argument("--hb_s", type=float, default=5.0)
    ap.add_argument("--dead_s", type=float, default=20.0)
    ap.add_argument("--gather_s", type=float, default=60.0)
    ap.add_argument("--found_wait_s", type=float, default=10.0)
    ap.add_argument("--regroup_min_s", type=float, default=30.0)
    ap.add_argument("--check_every_s", type=float, default=5.0)
    ap.add_argument("--ckpt_every_s", type=float, default=900.0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--chk_every", type=int, default=50)
    ap.add_argument("--init_from", default="")
    ap.add_argument("--iface", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--start_delay", type=float, default=0.0)
    ap.add_argument("--die_at_step", type=int, default=0)
    ap.add_argument("--hang_at_step", type=int, default=0)
    ap.add_argument("--hang_s", type=float, default=0.0)
    ap.add_argument("--leave_at_step", type=int, default=0)
    ap.add_argument("--hold_at_step", type=int, default=0, help="test hook: rank 0 sleeps --hold_s inside el.hold()")
    ap.add_argument("--hold_s", type=float, default=0.0)
    ap.add_argument("--step_sleep", type=float, default=0.0, help="test hook: sleep after each step (slow CPU run)")
    a = ap.parse_args(argv)
    if os.environ.get("ELASTIC_TRACE_AFTER"):              # debugging aid: dump all stacks after S seconds
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(float(os.environ["ELASTIC_TRACE_AFTER"]), repeat=False, file=sys.stderr)
    if a.start_delay:
        time.sleep(a.start_delay)
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    device = torch.device("cuda", local) if torch.cuda.is_available() and not a.cpu else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    train = torch.from_numpy(load_bytes(a.train).copy()).to(device)
    val = torch.from_numpy(load_bytes(a.val).copy()).to(device)
    out = run_elastic(a, device, train, val) if a.mode == "elastic" else run_ref(a, device, train, val)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
