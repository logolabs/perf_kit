"""perf_kit.t2i.fcdm.train: train or fine-tune the Agate FCDM text-to-image generator with the optimised loop.

    # fine-tune Agate Preview 002 on your own latents (one node)
    torchrun --nproc_per_node 4 -m perf_kit.t2i.fcdm.train --init Logolabs/agate-preview-002 --data shards/ \\
        --out runs/ft --global_batch 512 --micro_batch 128 --lr 2e-5 --warmup 500 --max_steps 5000 \\
        --compile --steer_upsample_once
    # from scratch: --preset agate (190.9M) --text ettin --text_path jhu-clsp/ettin-encoder-68m
    # throughput only: --synthetic_hw 32 (noise latents, generated captions)

The optimisations (each measured on GH200; WRITEUP.md, section "Agate"):
  --compile              torch.compile of the DDP-wrapped generator; captions padded to multiples of 32 tokens so
                         it sees few shapes. Checkpoints and the EMA use the uncompiled module (plain names).
  --steer_upsample_once  exact-maths change in the generator (model.FCDMThinker2._level): one upsampling of the
                         low-rank steering map per level instead of one per block of its 3C modulation maps.
  --ema_foreach          (default on) the EMA as one multi-tensor lerp; bit-identical, self-checked at step 1.
  --compile_text         (default on) the frozen text encoder's forward compiled in place (names unchanged).
  --ddp_bucket_view      (default on) DDP gradient_as_bucket_view: gradients are the all-reduce buckets, no copy.
  --ddp_static_graph     DDP static_graph; only with one micro-batch per step (it crashes DDP's reducer otherwise).
Model FLOPs for the logged MFU are counted once at start-up with torch's FLOP counter on the uncompiled model with
--steer_upsample_once off (the flag adds convolution FLOPs at full resolution while removing memory traffic; the
speed-up is judged by images per second, not by MFU).

Objective: rectified flow, t = 0 noise / t = 1 data, MSE on the velocity, logit-normal t, captions dropped to ""
with probability --cfg_drop (classifier-free guidance). t, the noise and the caption drop come from generators
seeded by (seed, step, rank, micro-batch), so --resume continues exactly. SIGTERM / SIGUSR1 stop at the next log
step: all ranks agree, rank 0 saves, main() returns 0. The final EMA is exported in Agate's release layout
(<out>/export/config.json + generator.safetensors) next to the text encoder it was trained with.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="perf_kit.t2i.fcdm.train")
    a = ap.add_argument
    a("--out", required=True)
    a("--init", default="", help="an Agate release (dir or HF repo: config.json + generator.safetensors) to start from")
    a("--preset", default="agate", choices=("agate", "tiny"), help="architecture when training from scratch")
    a("--text", default="", choices=("", "ettin", "t5", "hash"), help="default: ettin (hash with --preset tiny)")
    a("--text_path", default="", help="default: <release>/text_encoder, else jhu-clsp/ettin-encoder-68m")
    a("--text_max_len", type=int, default=128)
    a("--data", default="", help="DIR of latent shards (<name>.safetensors + <name>.jsonl), see data.py")
    a("--dataset", default="", help="module:function returning a dataset object (see data.py)")
    a("--synthetic_hw", type=int, default=0, help="throughput runs: noise latents of (4, N, N)")
    a("--vae_scale", type=float, default=0.18215)
    a("--global_batch", type=int, default=256)
    a("--micro_batch", type=int, default=128, help="images per GPU per forward/backward")
    a("--lr", type=float, default=1e-4)
    a("--betas", type=float, nargs=2, default=(0.9, 0.95))
    a("--weight_decay", type=float, default=0.0)
    a("--warmup", type=int, default=1000)
    a("--schedule", default="constant", choices=("constant", "cosine"))
    a("--lr_min", type=float, default=0.0)
    a("--grad_clip", type=float, default=1.0)
    a("--nan_guard", action=argparse.BooleanOptionalAction, default=True, help="skip a step with a non-finite grad")
    a("--ema_decay", type=float, default=0.9999)
    a("--ema_foreach", action=argparse.BooleanOptionalAction, default=True)
    a("--cfg_drop", type=float, default=0.1)
    a("--t_dist", default="logit_normal", choices=("logit_normal", "uniform"))
    a("--t_mu", type=float, default=0.0)
    a("--t_std", type=float, default=1.0)
    a("--compile", action="store_true")
    a("--compile_text", action=argparse.BooleanOptionalAction, default=True)
    a("--steer_upsample_once", action="store_true")
    a("--ddp_bucket_view", action=argparse.BooleanOptionalAction, default=True)
    a("--ddp_static_graph", action="store_true", help="DDP static_graph (one micro-batch per step only)")
    a("--channels_last", action="store_true")
    a("--epochs", type=int, default=1)
    a("--max_steps", type=int, default=0, help="0 = all epochs")
    a("--stop_after", type=int, default=0, help="stop at this step (the LR schedule still spans --max_steps)")
    a("--resume", action="store_true")
    a("--save_every", type=int, default=2000)
    a("--keep_last", type=int, default=3)
    a("--log_every", type=int, default=50)
    a("--seed", type=int, default=0)
    a("--device", default="cuda", choices=("cuda", "cpu"))
    return ap.parse_args(argv)


class StopFlag:
    def __init__(self):
        self.set, self.signum = False, None

    def install(self):
        for name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, self._handler)
                except ValueError:
                    pass
        return self

    def _handler(self, signum, frame):                # noqa: ARG002
        self.set, self.signum = True, signum


def lr_at(step: int, total: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if args.schedule == "constant":
        return args.lr
    q = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
    return args.lr_min + (args.lr - args.lr_min) * 0.5 * (1 + math.cos(math.pi * q))


def count_train_flops(model, hw: int, ctx_dim: int, L: int, dev) -> float:
    """Training FLOPs of ONE image (forward + backward, flow loss) counted on the uncompiled model with
    steer_upsample_once off; the RNG state is restored."""
    from torch.utils.flop_counter import FlopCounterMode
    from .flow import flow_loss
    once, model.steer_upsample_once = model.steer_upsample_once, False
    cpu_rng = torch.get_rng_state()
    try:
        n = 2
        x = torch.randn(n, 4, hw, hw, device=dev)
        ctx, mask = torch.randn(n, L, ctx_dim, device=dev), torch.ones(n, L, device=dev)
        t = torch.rand(n, device=dev)
        with FlopCounterMode(display=False) as fc:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                loss = flow_loss(model, x, t, ctx, mask, torch.randn_like(x))
            loss.backward()
        model.zero_grad(set_to_none=True)
        return fc.get_total_flops() / n
    finally:
        torch.set_rng_state(cpu_rng)
        model.steer_upsample_once = once


class Prefetch:
    """Background thread: the next steps' latents (pinned) and tokenized captions, in order."""

    def __init__(self, fn, start: int, end: int, depth: int = 2):
        self.q: queue.Queue = queue.Queue(depth)
        self.th = threading.Thread(target=self._run, args=(fn, start, end), daemon=True)
        self.th.start()

    def _run(self, fn, start, end):
        try:
            for s in range(start, end):
                self.q.put((s, fn(s)))
        except BaseException as e:                    # noqa: BLE001 -- surfaced in get()
            self.q.put((None, e))

    def get(self):
        s, item = self.q.get()
        if s is None:
            raise item
        return s, item


def save_ckpt(out: Path, step: int, model, ema, opt, args, cfg: dict, keep: int) -> Path:
    p = out / f"step-{step:07d}.pt"
    tmp = p.with_suffix(".pt.tmp")
    torch.save({"model": model.state_dict(), "ema": ema.shadow.state_dict(), "opt": opt.state_dict(), "step": step,
                "model_kw": cfg, "args": vars(args)}, tmp)
    os.replace(tmp, p)
    (out / "LATEST").write_text(p.name)
    if keep > 0:
        for old in sorted(out.glob("step-*.pt"))[:-keep]:
            old.unlink(missing_ok=True)
    return p


def export_release(out: Path, ema_model, model_kw: dict, args, step: int, text_src: str, hw: int) -> Path:
    """The EMA generator in Agate's release layout: config.json + generator.safetensors (fp32)."""
    from safetensors.torch import save_file
    d = out / "export"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in ema_model.state_dict().items()},
              str(d / "generator.safetensors"))
    cfg = {"model_type": "agate", "arch": "fcdm_t2", "model_kw": {k: list(v) if isinstance(v, tuple) else v
                                                                  for k, v in model_kw.items()},
           "text_encoder": text_src, "text_max_len": args.text_max_len, "vae": "stabilityai/sd-vae-ft-mse",
           "vae_scale": args.vae_scale, "latent_hw": hw, "resolution": 8 * hw, "sampler": {"kind": "euler", "steps": 50, "cfg": 3.0, "t0_is_noise": True,
                                                    "uncond_prompt": ""},
           "training": {"step": step, "trainer": "perf_kit.t2i.fcdm.train"}}
    (d / "config.json").write_text(json.dumps(cfg, indent=2))
    return d


def versions() -> dict:
    import platform
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "triton"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:                             # noqa: BLE001
            out[mod] = None
    from ... import __version__
    out["perf_kit"] = __version__
    return out


# =========================================================================== main
def main(argv=None) -> int:
    args = parse_args(argv)
    from .data import LatentShards, Synthetic, load_hook
    from .ema import EMA
    from .flow import flow_loss, sample_t
    from .model import FCDMThinker2, PRESETS, count_params, load_release
    from .text import build_text_encoder, compile_forward
    from ...telemetry import peak_bf16
    stop = StopFlag().install()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    cuda = args.device == "cuda"
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

    def log(rec):
        if rank == 0:
            print(json.dumps(rec), flush=True)
            with open(out / "log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    log({"event": "config", "args": vars(args), "world": world, "versions": versions(),
         "device": torch.cuda.get_device_name(dev) if cuda else "cpu"})
    if sum(bool(x) for x in (args.data, args.dataset, args.synthetic_hw)) != 1:
        raise SystemExit("give exactly one of --data, --dataset, --synthetic_hw")
    per_rank = args.global_batch // world
    if per_rank * world != args.global_batch or per_rank % args.micro_batch:
        raise SystemExit(f"--global_batch {args.global_batch} must split into {world} ranks x whole micro-batches of "
                         f"{args.micro_batch}")
    accum = per_rank // args.micro_batch
    if args.ddp_static_graph and accum > 1:            # DDP's reducer asserts on static_graph + no_sync accumulation
        raise SystemExit("--ddp_static_graph needs one micro-batch per step (--micro_batch = global_batch / ranks)")

    # ---- model + text ----------------------------------------------------------------------
    release = None
    torch.manual_seed(args.seed)                       # the from-scratch init is the same on every rank and run
    if args.init:
        model, rcfg, release = load_release(args.init)
        model_kw = rcfg["model_kw"]
    else:
        model_kw = dict(PRESETS[args.preset])
        model = FCDMThinker2(**model_kw)
    kind = args.text or ("hash" if args.preset == "tiny" and not args.init else "ettin")
    tpath = args.text_path or (str(release / "text_encoder") if release is not None else "jhu-clsp/ettin-encoder-68m")
    text = build_text_encoder(kind, tpath, dev, args.text_max_len, dim=model.config["ctx_dim"])
    if text.dim != model.config["ctx_dim"]:
        raise SystemExit(f"text encoder width {text.dim} != the generator's ctx_dim {model.config['ctx_dim']}")
    if args.compile_text and cuda and text.model is not None:
        compile_forward(text.model)
    model = model.to(dev)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.steer_upsample_once = bool(args.steer_upsample_once)
    model.train()
    ema = EMA(model, args.ema_decay, foreach=args.ema_foreach)
    if args.synthetic_hw:
        ds = Synthetic(args.synthetic_hw, args.seed)
    elif args.dataset:
        ds = load_hook(args.dataset, args)
    else:
        ds = LatentShards(args.data, args.global_batch, args.seed, args.vae_scale)
    x_probe, _ = ds.batch(0, 0, 0, world, 1)
    hw = x_probe.shape[-1]
    flops_img = count_train_flops(model, hw, model.config["ctx_dim"], min(args.text_max_len, 128), dev)
    peak = peak_bf16(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay,
                            fused=cuda)
    start = 0
    if args.resume and (out / "LATEST").exists():
        ck = torch.load(out / (out / "LATEST").read_text().strip(), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.shadow.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None,
                                                        gradient_as_bucket_view=args.ddp_bucket_view,
                                                        static_graph=args.ddp_static_graph)
    if args.compile:
        ddp = torch.compile(ddp)                       # checkpoints and EMA keep using the uncompiled `model`
    total = args.max_steps or args.epochs * ds.steps_per_epoch
    end = min(total, args.stop_after) if args.stop_after else total
    log({"event": "model", "params": count_params(model), "model_kw": {k: v for k, v in model_kw.items()},
         "init": args.init or None, "text": kind, "text_path": tpath, "text_dim": text.dim,
         "train_gflop_per_img": round(flops_img / 1e9, 2), "peak_tflops": peak / 1e12 if peak else None,
         "per_rank": per_rank, "micro_batch": args.micro_batch, "accum": accum, "steps": total, "start": start,
         "compile": args.compile, "compile_text": bool(args.compile_text and cuda), "ema_foreach": args.ema_foreach,
         "steer_upsample_once": args.steer_upsample_once, "ddp_bucket_view": args.ddp_bucket_view,
         "ddp_static_graph": args.ddp_static_graph})

    pad_to = 32 if args.compile else 1

    def fetch(step):
        epoch, s = divmod(step, ds.steps_per_epoch)
        x, caps = ds.batch(epoch, s, rank, world, per_rank)
        rnd = random.Random(((args.seed * 1_000_003 + step) * 4096 + rank) * 7 + 1)
        caps = ["" if rnd.random() < args.cfg_drop else c for c in caps]
        micro = []
        for j in range(accum):
            sl = slice(j * args.micro_batch, (j + 1) * args.micro_batch)
            ids, am = text.tokenize(caps[sl], pad_to)
            xm = x[sl].contiguous()
            micro.append((xm.pin_memory() if cuda else xm, ids, am))
        return micro

    pre = Prefetch(fetch, start, end)
    step, n_w, img_w, bad = start, 0, 0, 0
    loss_w = torch.zeros((), device=dev)
    ema_checked = not args.ema_foreach
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t_w = t_run = time.time()
    stopped = False
    gnorm = torch.zeros((), device=dev)
    while step < end:
        s, micro = pre.get()
        assert s == step
        model.loops = model.draw_loops(step, args.seed)
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, total, args)
        loss_step = torch.zeros((), device=dev)
        for j, (xm, ids, am) in enumerate(micro):
            g = torch.Generator().manual_seed(((args.seed * 1_000_003 + step) * 4096 + rank) * 64 + j)
            x1 = xm.to(dev, non_blocking=True).float()
            if args.channels_last:
                x1 = x1.contiguous(memory_format=torch.channels_last)
            ctx, mask = text.encode(ids, am)
            t = sample_t(x1.shape[0], dev, args.t_dist, args.t_mu, args.t_std, generator=g)
            noise = torch.randn(x1.shape, generator=g).to(dev, non_blocking=True)
            sync = ddp.no_sync() if (world > 1 and j < accum - 1 and hasattr(ddp, "no_sync")) else _null()
            with sync:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cuda):
                    loss = flow_loss(ddp, x1, t, ctx, mask, noise) / accum
                loss.backward()
            loss_step += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip if args.grad_clip > 0 else float("inf"))
        if args.nan_guard and not bool(torch.isfinite(gnorm)):       # one host sync per step (measured: free)
            opt.zero_grad(set_to_none=True)
            bad += 1
            log({"event": "skipped_step", "step": step, "reason": "non-finite gradient"})
            if bad >= 20:
                raise RuntimeError("20 consecutive non-finite steps")
        else:
            bad = 0
            opt.step()
            opt.zero_grad(set_to_none=True)
            if not ema_checked:
                n_chk, d_chk, n_skip = ema.check_foreach(model)
                log({"event": "ema_foreach_check", "tensors": n_chk, "skipped_frozen": n_skip, "max_abs_diff": d_chk})
                ema_checked = True
            ema.update(model)
        step += 1
        n_w += 1
        img_w += per_rank
        loss_w += loss_step
        if step % args.log_every == 0 or step == end or step == start + 1:
            if cuda:
                torch.cuda.synchronize(dev)
            el = time.time() - t_w
            rec = torch.stack([loss_w / n_w, torch.tensor(1.0 if stop.set else 0.0, device=dev)])
            if world > 1:
                dist.all_reduce(rec)
            lv, flag = rec.tolist()
            ips = img_w / max(el, 1e-9)
            log({"step": step, "loss": round(lv / world, 6), "lr": opt.param_groups[0]["lr"],
                 "grad_norm": round(float(gnorm), 4), "step_s": round(el / n_w, 4), "img_per_s_gpu": round(ips, 1),
                 "mfu": round(ips * flops_img / peak, 4) if peak else None,
                 "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None})
            n_w, img_w = 0, 0
            loss_w = torch.zeros((), device=dev)
            t_w = time.time()
            if flag > 0 and step < end:
                log({"event": "stop_signal", "step": step, "signal": stop.signum})
                stopped = True
        if rank == 0 and (stopped or step == end or (args.save_every and step % args.save_every == 0)):
            p = save_ckpt(out, step, model, ema, opt, args, model_kw, args.keep_last)
            log({"event": "saved", "path": str(p), "step": step})
            t_w = time.time()
        if stopped:
            break
    if rank == 0 and step >= total:
        d = export_release(out, ema.shadow, model_kw, args, step, tpath, hw)
        log({"event": "exported", "dir": str(d)})
    log({"event": "summary", "steps": step - start, "run_s": round(time.time() - t_run, 2), "stopped": stopped,
         "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
