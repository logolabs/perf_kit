"""perf_kit.dit.ideogram4.train: LoRA fine-tuning of Ideogram 4 with the fast DiT engine.

    torchrun --nproc_per_node 4 -m perf_kit.dit.ideogram4.train --weights ideogram-ai/ideogram-4-fp8 \\
        --data my_images/ --buckets 1024x1024,1024x768,768x1024 --out runs/r32 --rank 32 --alpha 32 \\
        --batch_size 4 --grad_accum 2 --lr 5e-5 --max_steps 2000 --compile --triton_swiglu
    (or one process per GPU under srun with RANK / WORLD_SIZE / LOCAL_RANK set)

Ideogram 4's weights are not part of perf_kit: they come from Ideogram's gated Hugging Face repository under the
Ideogram Non-Commercial Model Agreement, which you must accept first; an adapter trained on them is a Model
Derivative under that agreement. Install Ideogram's (Apache-2.0) inference package for the loaders:
`pip install git+https://github.com/ideogram-oss/ideogram4`.

Data: --data DIR with DIR/captions.jsonl ({"image": ..., "caption": ...} per line), or --dataset module:function
(precomputed latents + text features, or images + captions), or --synthetic (throughput only). See data.py.

Objective: flow matching in the package's convention (t = 0 noise, t = 1 data; z_t = (1 - t) noise + t x0,
v = x0 - noise), MSE on the velocity over image tokens; t from the package's resolution-aware logit-normal schedule
(--t_mu, --t_std; std 1.5 by default). t and the noise are drawn from a generator seeded by (seed, rank, step,
micro-batch), so a resumed run draws what an uninterrupted one would.

Engine options: --compile (torch.compile of the engine's elementwise functions; on GH200 the system g++ may be too
old for inductor's CPU wrapper, use a GCC >= 13 toolchain), --triton_swiglu, --split auto (each micro-batch's DiT
pass in the fewest sub-batches that need no block recompute; exact), --storage fp8|bf16, --fp8_forward (LOSSY,
measured no faster). Adapters are saved in PEFT format (one adapter) or the binned format (--n_bins > 1) with the
optimizer state, so --resume continues bit-for-bit in data order.

SIGTERM / SIGUSR1 stop at the next optimizer step: all ranks agree, rank 0 saves step-NNNNNNN, main() returns 0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

# useful FLOPs per image pixel for the logged MFU: Ideogram 4's LoRA training step counted with torch's FLOP counter
# on the reference trainer (95.16 TFLOP for one 1024 x 448 sample, internal measurement 2026-10-31); every MFU
# perf_kit reports for Ideogram 4 uses this one number for both the reference and the engine
FLOP_PX = 95.16e12 / (1024 * 448)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="perf_kit.dit.ideogram4.train")
    a = ap.add_argument
    a("--weights", default="ideogram-ai/ideogram-4-fp8",
      help="Hugging Face repo of the Ideogram 4 weights (read through the HF cache; HF_HUB_OFFLINE=1 once cached)")
    a("--out", required=True)
    a("--data", default="", help="DIR with captions.jsonl (image + caption per line)")
    a("--dataset", default="", help="module:function returning a dataset object (see data.py)")
    a("--synthetic", action="store_true", help="random latents + text features (throughput only)")
    a("--tiny", action="store_true", help="a toy transformer from the vendored definition (CPU tests)")
    a("--buckets", default="1024x1024", help="canvas sizes WxH (multiples of 16), nearest aspect ratio wins")
    a("--limit", type=int, default=0)
    a("--max_text_tokens", type=int, default=2048)
    a("--synthetic_text_len", type=int, default=400)
    a("--rank", type=int, default=32)
    a("--alpha", type=float, default=32.0)
    a("--rslora", action=argparse.BooleanOptionalAction, default=True, help="scale alpha / sqrt(r) (else alpha / r)")
    a("--n_bins", type=int, default=1, help="timestep-binned adapters, soft-blended per sample (1 = one LoRA)")
    a("--base_bins", type=int, default=0, help="with --n_bins: lowest-noise bins without an adapter")
    a("--lr", type=float, default=5e-5)
    a("--loraplus", type=float, default=1.0, help="lr multiplier of the B matrices")
    a("--weight_decay", type=float, default=0.01)
    a("--warmup", type=int, default=100)
    a("--schedule", default="cosine", choices=("cosine", "constant"))
    a("--min_lr_ratio", type=float, default=0.0)
    a("--grad_clip", type=float, default=1.0)
    a("--batch_size", type=int, default=4, help="samples per micro-batch per GPU")
    a("--grad_accum", type=int, default=1)
    a("--epochs", type=int, default=1)
    a("--max_steps", type=int, default=0, help="0 = all epochs")
    a("--stop_after", type=int, default=0, help="stop at this step (the LR schedule still spans --max_steps)")
    a("--t_mu", type=float, default=0.0)
    a("--t_std", type=float, default=1.5)
    a("--storage", default="fp8", choices=("fp8", "bf16"))
    a("--compile", action="store_true")
    a("--triton_swiglu", action="store_true")
    a("--split", default="auto", help="auto = fewest sub-batches with no block recompute; N = N sub-batches")
    a("--act_budget_gb", type=float, default=0.0, help="activation budget (0 = free memory after setup - reserve)")
    a("--reserve_gb", type=float, default=10.0)
    a("--fp8_forward", action="store_true", help="LOSSY FP8 forward GEMMs (measured: no speed-up, 10x error)")
    a("--init_adapter", default="", help="a saved step dir (adapter_model.safetensors) to start from")
    a("--resume", action="store_true", help="continue from <out>/LATEST")
    a("--save_every", type=int, default=500)
    a("--log_every", type=int, default=10)
    a("--flop_px", type=float, default=FLOP_PX)
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


# --------------------------------------------------------------------------- optimizer: two flat parameters
def make_optimizer(eng, lr: float, wd: float, loraplus: float, fused: bool):
    """AdamW (betas 0.9 / 0.99, eps 1e-8) over two flat parameters that ARE the LoRA store's A and B halves."""
    L = eng.lora
    pa = torch.nn.Parameter(L.master[: L.n_a], requires_grad=True)
    pb = torch.nn.Parameter(L.master[L.n_a:], requires_grad=True)
    assert pa.data_ptr() == L.master.data_ptr()
    pa.grad, pb.grad = L.grad[: L.n_a], L.grad[L.n_a:]
    groups = [{"params": [pa], "lr": lr, "weight_decay": wd, "lr_mult": 1.0},
              {"params": [pb], "lr": lr * loraplus, "weight_decay": wd, "lr_mult": loraplus}]
    return torch.optim.AdamW(groups, betas=(0.9, 0.99), eps=1e-8, fused=fused), pa, pb


def lr_at(step: int, total: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    if args.schedule == "constant":
        return args.lr
    q = min(1.0, (step - args.warmup) / max(1, total - args.warmup))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * q)))


# --------------------------------------------------------------------------- memory plan
def n_save_for(eng, T: int, budget_bytes: float) -> int:
    """Blocks whose lean activations fit: the float32 residual stream (n_layers x T x D x 4) is always kept; each
    saved block adds T x saved_bytes_per_token; transients ~T x 0.4 MB."""
    nl = eng.n_layers
    if budget_bytes == float("inf"):
        return nl
    resid = nl * T * eng.D * 4
    per = T * eng.saved_bytes_per_token()
    return max(0, min(nl, int((budget_bytes - resid - T * 400_000) // max(1, per))))


def plan_split(eng, lens: list, n_img: int, budget_bytes: float, split: str) -> list:
    """Sample-index chunks for the DiT pass of one micro-batch. Layout only: t, noise, text features and latents
    are drawn / computed for the whole micro-batch and each chunk is weighted len(chunk) / B, so the gradient is the
    micro-batch's (tested). auto: the fewest chunks whose every block fits the activation budget (no recompute)."""
    B = len(lens)

    def chunks(k):
        size = -(-B // k)
        return [list(range(i, min(B, i + size))) for i in range(0, B, size)]
    if split != "auto":
        return chunks(max(1, min(B, int(split))))
    for k in range(1, B + 1):
        c = chunks(k)
        T = max(sum(lens[j] + n_img for j in ch) for ch in c)
        if n_save_for(eng, T, budget_bytes) >= eng.n_layers:
            return c
    return chunks(B)


def train_split(eng, lens, gh, gw, dev, llm, z_t, v, t, loss_scale, budget, split):
    """One micro-batch through the engine in sub-batches. llm (B, mt, D_llm) left-padded prefix features, z_t / v
    (B, n_img, C). Returns (micro-batch loss = mean over its samples, packed tokens, min n_save)."""
    from .engine import pack_layout
    B = len(lens)
    n_img = gh * gw
    mt = max(lens)
    tot = 0.0
    T_all = 0
    ns_min = eng.n_layers
    for ch in plan_split(eng, lens, n_img, budget, split):
        sub = [lens[j] for j in ch]
        p = pack_layout(sub, gh, gw, dev)
        idx = torch.tensor(ch, device=llm.device) if len(ch) != B else None
        l_ = llm if idx is None else llm.index_select(0, idx)
        if p.mt != mt:                                    # this chunk's longest caption is shorter: re-left-pad
            l_ = l_[:, mt - p.mt:]
        zt_ = z_t if idx is None else z_t.index_select(0, idx)
        v_ = v if idx is None else v.index_select(0, idx)
        t_ = t if idx is None else t.index_select(0, idx)
        llm_text = l_.reshape(len(ch) * p.mt, -1)[p.text_src_prefix]
        ns = n_save_for(eng, p.T, budget)
        ns_min = min(ns_min, ns)
        loss, _ = eng.train_micro(p, llm_text, zt_.reshape(len(ch) * n_img, -1), v_.reshape(len(ch) * n_img, -1),
                                  t_, loss_scale * len(ch) / B, n_save=ns)
        tot = tot + loss * (len(ch) / B)
        T_all += p.T
    return tot, T_all, ns_min


def left_pad(feats: list, device) -> torch.Tensor:
    """[(n_i, D)] -> (B, max n, D) left-padded with zeros (the package's text layout)."""
    mt = max(f.shape[0] for f in feats)
    out = torch.zeros(len(feats), mt, feats[0].shape[1], dtype=torch.float32)
    for b, f in enumerate(feats):
        out[b, mt - f.shape[0]:] = f.float()
    return out.to(device)


def schedule_t(u: torch.Tensor, hw: tuple[int, int], mu: float, std: float) -> torch.Tensor:
    from .upstream.scheduler import get_schedule_for_resolution
    return get_schedule_for_resolution(hw, known_mean=mu, std=std)(u)


# --------------------------------------------------------------------------- checkpoints
def adapter_config(args, eng) -> dict:
    if eng.K == 1 and args.base_bins == 0:
        return {"peft_type": "LORA", "r": args.rank, "lora_alpha": args.alpha, "lora_dropout": 0.0, "bias": "none",
                "target_modules": ["qkv", "o", "w1", "w2", "w3"], "use_rslora": bool(args.rslora),
                "use_dora": False, "fan_in_fan_out": False, "inference_mode": True,
                "base_model_name_or_path": args.weights}
    return {"kind": "binned_lora", "n_bins": args.n_bins, "base_bins": args.base_bins, "rank": args.rank,
            "alpha": args.alpha, "rslora": bool(args.rslora), "targets": list(_targets())}


def _targets():
    from .engine import TARGETS
    return TARGETS


def save_step(out: Path, step: int, eng, opt, pa, pb, meta: dict, args) -> Path:
    from safetensors.torch import save_file
    d = out / f"step-{step:07d}"
    tmp = out / f".tmp-{step:07d}"
    tmp.mkdir(parents=True, exist_ok=True)
    tensors = eng.lora.state_peft() if (eng.K == 1 and args.base_bins == 0) else eng.lora.state_binned()
    save_file(tensors, str(tmp / "adapter_model.safetensors"), metadata={"format": "pt"})
    (tmp / "adapter_config.json").write_text(json.dumps(adapter_config(args, eng), indent=2))
    st = {"step": step, "meta": meta, "groups": [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups],
          "opt": [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in opt.state[p].items()} for p in (pa, pb)]}
    torch.save(st, tmp / "state.pt")
    if d.exists():
        import shutil
        shutil.rmtree(d)
    os.replace(tmp, d)
    (out / "LATEST").write_text(d.name)
    return d


def load_adapter_dir(eng, d: Path) -> None:
    from safetensors.torch import load_file
    sd = load_file(str(d / "adapter_model.safetensors"))
    if any(k.endswith(".lora_A.weight") for k in sd):
        eng.lora.load_peft(sd)
    else:
        eng.lora.load_binned(sd)


def load_state(opt, pa, pb, d: Path) -> dict:
    st = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
    for g, saved in zip(opt.param_groups, st["groups"]):
        g.update({k: v for k, v in saved.items() if k in g})
    fused = bool(opt.defaults.get("fused"))
    for p, s in zip((pa, pb), st["opt"]):              # fused AdamW keeps `step` on the device, torch's loop on the host
        opt.state[p] = {k: (v.to(p.device) if torch.is_tensor(v) and (v.dim() > 0 or fused) else v)
                        for k, v in s.items()}
    return st


def versions() -> dict:
    import platform
    out = {"python": platform.python_version()}
    for mod in ("torch", "triton", "flash_attn_interface", "ideogram4"):
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
    from . import ops
    from .data import ImageCaptionFolder, Synthetic, load_hook, parse_buckets
    from .engine import FastDiT
    from .stack import Ideogram4Stack, tiny_transformer
    stop = StopFlag().install()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    cuda = args.device == "cuda"
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32 = True
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
    buckets = parse_buckets(args.buckets)
    if sum(bool(x) for x in (args.data, args.dataset, args.synthetic)) != 1:
        raise SystemExit("give exactly one of --data, --dataset, --synthetic")
    if args.tiny and not (args.synthetic or args.dataset):
        raise SystemExit("--tiny has no text encoder or VAE: use it with --synthetic or a --dataset of features")

    # ---- model ----------------------------------------------------------------------------
    t0 = time.time()
    stack = None
    if args.tiny:
        cdt = torch.bfloat16 if cuda else torch.float32
        tr = tiny_transformer(dtype=cdt, device=dev)
        storage = "bf16"
    else:
        need_enc = bool(args.data) or bool(args.dataset)
        stack = Ideogram4Stack(args.weights, dev, torch.bfloat16, args.max_text_tokens, text=need_enc, vae=need_enc)
        tr, cdt, storage = stack.transformer, torch.bfloat16, args.storage
    eng = FastDiT(tr, rank=args.rank, alpha=args.alpha, rslora=args.rslora, n_bins=args.n_bins,
                  base_bins=args.base_bins, compute_dtype=cdt, master_dtype=torch.float32, storage=storage,
                  attn="auto" if cuda else "ref", fp8_forward=args.fp8_forward)
    ops.set_compile(args.compile)
    ops.set_triton_swiglu(args.triton_swiglu)
    if args.synthetic:
        ds = Synthetic(buckets, tr.config.llm_features_dim, args.synthetic_text_len, args.seed)
    elif args.dataset:
        ds = load_hook(args.dataset, args)
    else:
        ds = ImageCaptionFolder(args.data, buckets, args.seed, args.limit)
    opt, pa, pb = make_optimizer(eng, args.lr, args.weight_decay, args.loraplus, fused=cuda)
    start_step, skip = 0, 0
    if args.resume and (out / "LATEST").exists():
        ck = out / (out / "LATEST").read_text().strip()
        load_adapter_dir(eng, ck)
        st = load_state(opt, pa, pb, ck)
        start_step = st["step"]
    elif args.init_adapter:
        load_adapter_dir(eng, Path(args.init_adapter))
    else:
        eng.lora.init_new(args.seed)
    if cuda:
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(dev)
        budget = args.act_budget_gb * 1e9 if args.act_budget_gb > 0 else free - args.reserve_gb * 1e9
    else:
        budget = float("inf")
    log({"event": "model", "tiny": args.tiny, "storage": storage, "frozen_target_gb": round(eng.frozen_bytes() / 1e9, 3),
         "attn": eng.attn, "compile": args.compile, "triton_swiglu": args.triton_swiglu, "split": args.split,
         "fp8_forward": args.fp8_forward, "adapters": eng.K, "lora_rank": eng.rank, "lora_scale": eng.scale,
         "lora_params": eng.lora.numel, "act_budget_gb": round(budget / 1e9, 2) if budget != float("inf") else None,
         "saved_kb_per_token_block": round(eng.saved_bytes_per_token() / 1e3, 2), "start_step": start_step,
         "secs": round(time.time() - t0, 1)})

    # ---- schedule ---------------------------------------------------------------------------
    def epoch_len():
        if isinstance(ds, ImageCaptionFolder):
            return len(ds.epoch_batches(0, args.batch_size)) // world
        return getattr(ds, "n", None) or getattr(ds, "micro_per_epoch", lambda: 0)()
    per_epoch = epoch_len()
    steps_per_epoch = max(1, per_epoch // args.grad_accum) if per_epoch else 0
    total = args.max_steps or args.epochs * steps_per_epoch
    if total <= 0:
        raise SystemExit("set --max_steps (the dataset does not say how long an epoch is)")
    end = min(total, args.stop_after) if args.stop_after else total
    loss_scale = 1.0 / (args.grad_accum * world)

    # ---- loop --------------------------------------------------------------------------------
    step, micro_done = start_step, start_step * args.grad_accum
    epoch = micro_done // per_epoch if per_epoch else 0
    skip = micro_done - epoch * per_epoch if per_epoch else micro_done
    losses, px_w, tok_w, n_w, ns_hist = [], 0, 0, 0, []
    step_loss = 0.0
    gnorm = torch.zeros((), device=dev)
    from ...telemetry import peak_bf16
    peak = peak_bf16(dev)
    eng.lora.zero_grad()
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t_w = t_run = time.time()
    stopped, k = False, 0
    while step < end and not stopped:
        for mb in ds.batches(epoch, rank, world, args.batch_size):
            if skip:
                skip -= 1
                continue
            g = torch.Generator(device="cpu").manual_seed(((args.seed * 1_000_003 + rank) * 1_000_003 + step) * 64 + k)
            if "pixels" in mb:
                pix = mb["pixels"]
                gh, gw = pix.shape[-2] // 16, pix.shape[-1] // 16
                toks = stack.tokenize(mb["captions"])
                lens = [n for _, n in toks]
                llm = stack.encode_text_prefix(toks)
                x0 = stack.encode_images(pix)
            else:
                gh, gw = mb["grid"]
                lens = [f.shape[0] for f in mb["text_features"]]
                llm = left_pad(mb["text_features"], dev)
                x0 = mb["latents"].to(dev, torch.float32)
            B = x0.shape[0]
            u = torch.rand(B, generator=g).clamp(1e-6, 1 - 1e-6)
            t = schedule_t(u, (gh * 16, gw * 16), args.t_mu, args.t_std).to(dev)
            noise = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(dev)
            tb = t.view(-1, 1, 1).to(x0.dtype)
            z_t = (1.0 - tb) * noise + tb * x0
            v = x0 - noise
            loss, T_mb, ns = train_split(eng, lens, gh, gw, dev, llm, z_t, v, t, loss_scale, budget, args.split)
            step_loss = step_loss + loss / args.grad_accum
            ns_hist.append(ns)
            px_w += B * gh * gw * 256
            tok_w += T_mb
            k += 1
            if k < args.grad_accum:
                continue
            k = 0
            # ---- optimizer step: one all-reduce, clip on device, fused AdamW, refresh compute copies
            if world > 1:
                dist.all_reduce(eng.lora.grad)
            gnorm = eng.lora.grad.norm()
            if args.grad_clip > 0:
                eng.lora.grad.mul_((args.grad_clip / (gnorm + 1e-6)).clamp(max=1.0))
            lr = lr_at(step, total, args)
            for grp in opt.param_groups:
                grp["lr"] = lr * grp["lr_mult"]
            opt.step()
            eng.lora.zero_grad()
            eng.lora.push()
            step += 1
            n_w += 1
            losses.append(step_loss.detach())
            step_loss = 0.0
            if step % args.log_every == 0 or step == end or step == start_step + 1:
                if cuda:
                    torch.cuda.synchronize(dev)
                el = time.time() - t_w
                rec = torch.stack([torch.stack(losses).float().mean().to(dev),
                                   torch.tensor(1.0 if stop.set else 0.0, device=dev)])
                if world > 1:
                    dist.all_reduce(rec)
                lv, flag = rec.tolist()
                log({"step": step, "loss": round(lv / world, 6), "lr": lr, "grad_norm": round(float(gnorm), 5),
                     "step_s": round(el / n_w, 4), "img_px_per_s_gpu": round(px_w / max(el, 1e-9)),
                     "tokens_per_s_gpu": round(tok_w / max(el, 1e-9)),
                     "mfu": round(args.flop_px * px_w / el / peak, 4) if peak else None,
                     "n_save_min": min(ns_hist), "epoch": epoch,
                     "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None})
                losses, px_w, tok_w, n_w, ns_hist = [], 0, 0, 0, []
                t_w = time.time()
                if flag > 0 and step < end:
                    log({"event": "stop_signal", "step": step, "signal": stop.signum})
                    stopped = True
            if rank == 0 and (stopped or (args.save_every and step % args.save_every == 0) or step == end):
                d = save_step(out, step, eng, opt, pa, pb, {"epoch": epoch}, args)
                log({"event": "saved", "dir": str(d), "step": step})
                t_w = time.time()
            if stopped or step >= end:
                break
        else:
            epoch += 1
            continue
        break
    log({"event": "summary", "steps": step - start_step, "run_s": round(time.time() - t_run, 2), "stopped": stopped,
         "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
