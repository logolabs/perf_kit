"""perf_kit.train: generic LoRA SFT on chat / prompt-completion JSONL with the perf_kit engine.

    torchrun --nproc_per_node 4 -m perf_kit.train --model <hf dir> --train data.jsonl --val val.jsonl --out run/ \
        --rank 16 --alpha 32 --lr 1e-4 --global_batch 32 --micro_tokens 4096
    (or one process per GPU under srun with RANK / WORLD_SIZE / LOCAL_RANK set)

Supported checkpoints: Qwen3.5 / 3.6 / 3.8 hybrids (models/qwen3_5.py) and Llama / Mistral / Qwen2 / Qwen3 dense
decoders (models/dense.py), bf16 or block-FP8 safetensors; VLM checkpoints train their text decoder. Adapters are
saved in PEFT format (loadable by peft / vLLM); the optimizer state is saved next to them for exact resumption.

Step sets: `--step_order bucketed` (default) uses a length-bucketed sampler (--ref_micro_bs rows per micro-batch,
interleaved over --ref_world ranks), so a run is reproducible on any number of GPUs; `--step_order sequential` takes
--global_batch consecutive rows per step (what an unshuffled sampler in another framework sees). Either way the rows
of a step are dealt to ranks by LPT and packed without padding, and the loss is normalised by the step's global
label count, so the gradient does not depend on the layout.

Memory: the per-micro-batch recompute plan needs the stored-activation size per token per layer; by default
(--act_mb 0) it is measured on the real model at start-up (one probe forward), see engine.measure_act_mb.

Stopping: SIGTERM or SIGUSR1 stops at the next log-step boundary (all ranks agree), saves step-NNNNNN (+ optimizer
state with --optim_dir) and returns 0, so a scheduler's grace signal never loses work and wrappers that need a
normal return (energy monitors) still record.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import time
from pathlib import Path

PEAK_BF16 = 989e12          # H100 / GH200 SXM dense bf16 peak, the MFU denominator


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="perf_kit.train")
    a = ap.add_argument
    a("--model", required=True)
    a("--tokenizer", default="")
    a("--train", nargs="+", required=True)
    a("--val", default="")
    a("--out", required=True)
    a("--chat_template_kwargs", default="", help='JSON passed to the chat template, e.g. \'{"enable_thinking": false}\'')
    a("--rank", type=int, default=16)
    a("--alpha", type=float, default=32)
    a("--targets", nargs="*", default=None, help="LoRA target module names (default: every projection)")
    a("--lr", type=float, default=1e-4)
    a("--warmup", type=int, default=50)
    a("--min_lr_ratio", type=float, default=0.1)
    a("--epochs", type=float, default=1.0)
    a("--max_steps", type=int, default=0)
    a("--global_batch", type=int, default=32)
    a("--step_order", default="bucketed", choices=("bucketed", "sequential"))
    a("--ref_micro_bs", type=int, default=4)
    a("--ref_world", type=int, default=0)
    a("--micro_tokens", type=int, default=4096)
    a("--max_len", type=int, default=4096)
    a("--limit", type=int, default=0)
    a("--val_rows", type=int, default=256)
    a("--eval_every", type=int, default=200)
    a("--save_every", type=int, default=0)
    a("--no_save", action="store_true", help="never write adapters (benchmarks)")
    a("--log_every", type=int, default=10)
    a("--seed", type=int, default=0)
    a("--storage", default="fp8", choices=("fp8", "bf16"))
    a("--ckpt", default="auto", choices=("auto", "all", "none"))
    a("--act_mb", type=float, default=0.0, help="MiB per token per layer for the recompute plan (0 = measure)")
    a("--act_probe_tokens", type=int, default=2048)
    a("--mem_frac", type=float, default=0.92)
    a("--reserve_gb", type=float, default=6.0)
    a("--act_offload", action="store_true", help="EXPERIMENTAL: activations to pinned host memory (exact)")
    a("--offload_keep_last", type=int, default=4)
    a("--head_chunk", type=int, default=1024)
    a("--token_cache", default="")
    a("--encode_procs", type=int, default=0)
    a("--init_adapter", default="")
    a("--start_step", type=int, default=0)
    a("--stop_at", type=int, default=0, help="end before this step (saves step-NNNNNN + optimizer to resume later)")
    a("--optim_dir", default="")
    a("--resume_optim", action="store_true")
    a("--mfu_params", type=float, default=0.0, help="N of the logged 4N-per-token MFU (0 = frozen projections + head)")
    a("--profile_at", type=int, default=0)
    a("--wandb_project", default="", help="log to W&B (empty = off)")
    a("--impl", default="auto", choices=("auto", "fast", "ref"))
    a("--device", default="cuda", choices=("cuda", "cpu"))
    return ap.parse_args(argv)


def load_tokenizer(src):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(src)


def build_model(args, dev, dtype):
    from transformers import AutoConfig
    from .weights import SafetensorsSource
    cfg = AutoConfig.from_pretrained(args.model)
    src = SafetensorsSource(args.model)
    names = src.names()
    prefix = "model.language_model." if any(n.startswith("model.language_model.") for n in names) else "model."
    mt = getattr(getattr(cfg, "text_config", cfg), "model_type", "")
    kw = dict(prefix=prefix, rank=args.rank, alpha=args.alpha, storage=args.storage, device=dev, dtype=dtype,
              impl=args.impl)
    if args.targets:
        kw["targets"] = tuple(args.targets)
    if mt.startswith("qwen3_5"):
        from .models.qwen3_5 import FastQwen35
        m = FastQwen35(cfg, src, **kw)
    elif mt in ("llama", "mistral", "qwen2", "qwen3"):
        from .models.dense import FastDense
        m = FastDense(cfg, src, **kw)
    else:
        raise SystemExit(f"perf_kit has no fast model for model_type {mt!r}")
    src.close()
    return m


class StopFlag:
    """SIGTERM / SIGUSR1 -> a flag only; the training loop acts on it at a step boundary."""

    def __init__(self):
        self.set = False
        self.signum = None

    def install(self):
        for name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, self._handler)
                except ValueError:                    # not the main thread
                    pass
        return self

    def _handler(self, signum, frame):                # noqa: ARG002
        self.set, self.signum = True, signum


def versions() -> dict:
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for mod in ("torch", "transformers", "triton", "flash_attn", "fla", "liger_kernel"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:                             # noqa: BLE001
            out[mod] = None
    from . import __version__
    out["perf_kit"] = __version__
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    import torch.distributed as dist
    from . import data as D
    from .engine import CkptPolicy, Engine, Prefetcher, lr_cosine, measure_act_mb
    from .kernels import ops
    from .layout import SequentialSteps, plan_step, reference_steps, split_micro
    from .packing import pack
    from .weights import load_adapter, save_adapter
    stop = StopFlag().install()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    cuda = args.device == "cuda"
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cuda.enable_cudnn_sdp(False)     # cuDNN SDPA returned NaNs on GH200 (torch 2.13)
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    dtype = torch.bfloat16 if cuda else torch.float32
    torch.manual_seed(args.seed)
    tkw = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    wb = None
    if rank == 0 and args.wandb_project and not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=out.name, config=vars(args))
        except Exception as ex:                       # noqa: BLE001
            print(f"wandb unavailable: {ex!r}"[:200])

    def log(rec):
        if rank != 0:
            return
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if wb is not None:
            wb.log({f"{rec.get('event', 'train')}/{k}": v for k, v in rec.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})

    def barrier():
        if world > 1:
            dist.barrier()

    log({"event": "config", "args": vars(args), "world": world, "versions": versions(),
         "device": torch.cuda.get_device_name(dev) if cuda else "cpu", "kernels": ops.available()})
    tok = load_tokenizer(args.tokenizer or args.model)
    t0 = time.time()
    tsrc = args.tokenizer or args.model
    if rank == 0 or not args.token_cache:
        rows, dropped, src = D.load(args.train, tok, tsrc, args.max_len, args.limit, args.token_cache,
                                    args.encode_procs, tkw)
    barrier()
    if rank != 0 and args.token_cache:
        rows, dropped, src = D.load(args.train, tok, tsrc, args.max_len, args.limit, args.token_cache, 0, tkw)
    val = []
    if args.val:
        vr, _, _ = D.load([args.val], tok, tsrc, args.max_len, args.val_rows, "", 0, tkw)
        val = [vr[i] for i in range(len(vr))]
    log({"event": "data", "rows": len(rows), "dropped_long": dropped, "tokens": int(rows.off[-1]), "val": len(val),
         "source": src, "secs": round(time.time() - t0, 1), "world": world})

    t0 = time.time()
    model = build_model(args, dev, dtype)
    acfg = None
    if args.init_adapter:
        tensors, acfg = load_adapter(args.init_adapter)
        model.lora.load_peft(tensors)
    else:
        model.lora.init_new(args.seed)
    acfg = dict(acfg or {"peft_type": "LORA", "r": args.rank, "lora_alpha": args.alpha, "lora_dropout": 0.0,
                         "bias": "none", "target_modules": sorted(model.targets), "base_model_name_or_path": args.model,
                         "task_type": "CAUSAL_LM", "fan_in_fan_out": False, "use_rslora": False, "use_dora": False})
    acfg["inference_mode"] = True
    total_mem = torch.cuda.get_device_properties(dev).total_memory / 2**30 if cuda else 0.0
    static = torch.cuda.memory_allocated(dev) / 2**30 if cuda else 0.0
    adam = 2 * model.lora.master.numel() * 4 / 2**30
    budget = max(0.0, args.mem_frac * total_mem - static - adam - args.reserve_gb) if cuda else 1e9
    policy = CkptPolicy(len(model.layers), budget, mode=args.ckpt)
    lengths = [rows.length(i) for i in range(len(rows))]
    n_labels = [rows.n_labels(i) for i in range(len(rows))]
    calib = None
    if args.act_mb > 0:
        policy.act_mb = args.act_mb
    elif cuda and args.ckpt == "auto":
        probe, n_tok = [], 0                         # the first rows, up to ~act_probe_tokens tokens
        for i in range(len(rows)):
            if probe and n_tok + lengths[i] > args.act_probe_tokens:
                break
            probe.append(rows[i])
            n_tok += lengths[i]
        calib = measure_act_mb(model, pack(probe).to(dev))
        policy.calibrate(calib["act_mb"], calib["in_mb"])
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    offload = None
    if args.act_offload:
        from .offload import ActOffload
        offload = ActOffload(len(model.layers), keep_last=args.offload_keep_last)
    eng = Engine(model, lr=args.lr, head_chunk=args.head_chunk, world=world, dist=dist if world > 1 else None,
                 ckpt=policy, segs=(2,), act_offload=offload)
    optim_file = Path(args.optim_dir).expanduser() / "optim.pt" if args.optim_dir else None
    if args.resume_optim:
        meta = json.loads(optim_file.with_suffix(".json").read_text())
        if meta["step"] != args.start_step:
            raise SystemExit(f"optimizer state is at step {meta['step']}, --start_step is {args.start_step}")
        eng.load_optim_state(torch.load(optim_file, map_location=dev, weights_only=True))
    n_lin, n_head, n_lora = model.frozen_linear_params(), model.head_params(), model.lora.numel
    n_mfu = args.mfu_params or (n_lin + n_head)
    log({"event": "model", "class": type(model).__name__, "frozen_linear_params": n_lin, "head_params": n_head,
         "lora_params": n_lora, "lora_A_abs_sum": float(model.lora.master[: model.lora.n_a].double().abs().sum()),
         "n_layers": len(model.layers), "storage": args.storage,
         "static_gb": round(static, 2), "act_budget_gb": round(budget, 2), "act_calibration": calib,
         "ckpt_act_mb": round(policy.act_mb, 4), "ckpt_in_mb": round(policy.in_mb, 4),
         "tokens_without_recompute": policy.max_tokens_without_recompute() if cuda else None,
         "act_offload": bool(offload), "secs": round(time.time() - t0, 1)})

    ref_world = args.ref_world or world
    if args.step_order == "sequential":
        srcs = SequentialSteps(len(rows), args.global_batch)
    else:
        accum = max(1, args.global_batch // ref_world // args.ref_micro_bs)
        srcs = reference_steps(lengths, args.ref_micro_bs, ref_world, accum, args.seed)
    total = args.max_steps or max(1, int(srcs.steps_per_epoch * args.epochs))
    pre = Prefetcher(lambda i: rows[i], dev)
    vpre = Prefetcher(lambda i: val[i], dev)
    vmicro = split_micro(list(range(rank, len(val), world)), [len(v[0]) for v in val], args.micro_tokens)

    def run_val():
        a = eng.eval_ce(vpre.run(vmicro)).tolist()
        return round(a[0] / max(1, a[1]), 6), int(a[1])

    def save(tag, step_):
        if rank != 0 or args.no_save:
            return
        save_adapter(str(out / tag), model.lora.state_peft(), acfg)
        if optim_file is not None:
            optim_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(eng.optim_state(), optim_file)
            optim_file.with_suffix(".json").write_text(json.dumps({"step": step_, "adapter": str(out / tag)}))
        log({"event": "saved", "dir": str(out / tag), "step": step_})

    def sync():
        if cuda:
            torch.cuda.synchronize(dev)

    def do_val(step_):
        vl, vn = run_val()
        log({"event": "val", "step": step_, "val_loss": vl, "val_targets": vn})

    if val:
        do_val(args.start_step)
    step, tok_w, lab_w, n_w, rows_w = args.start_step, 0, 0, 0, 0
    hist: list = []                                    # per-step [loss sum, target count] (GPU) since the last log
    sync()
    t_w = t_run = time.time()
    prof = None
    stopped = False
    end = min(total, args.stop_at) if args.stop_at else total
    while step < end:
        if args.profile_at and step == args.profile_at and cuda:
            from torch.profiler import ProfilerActivity, profile
            sync()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.__enter__()
            t_p = time.time()
        lay = plan_step(srcs.step(step), lengths, n_labels, rank, world, args.micro_tokens)
        lr = lr_cosine(step, total, args.lr, args.warmup, args.min_lr_ratio)
        gn = eng.train_step(lay, pre.run(lay.micro), lr)
        hist.append(eng.acc_step[:2].clone())          # async copy: no host sync
        step += 1
        n_w += 1
        tok_w += lay.tokens_total
        lab_w += lay.label_total
        rows_w += len(lay.rows)
        if prof is not None and step == args.profile_at + 2:
            sync()
            prof.__exit__(None, None, None)
            from .telemetry import summarize
            log({"event": "profile", **summarize(prof, 2, time.time() - t_p)})
            prof = None
        if step % args.log_every == 0 or step == args.start_step + 1 or step == end:
            sync()
            el = time.time() - t_w
            acc = eng.acc.clone()
            H = torch.stack(hist)
            flag = torch.tensor([1.0 if stop.set else 0.0], dtype=acc.dtype, device=acc.device)
            if world > 1:
                dist.all_reduce(acc)
                dist.all_reduce(H)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            a = acc.tolist()
            eng.acc.zero_()
            hl = H.tolist()
            tps = tok_w / world / max(el, 1e-9)
            ck = eng.ckpt_layers / max(1, eng.ckpt_total)
            log({"step": step, "loss": round(a[0] / max(1, a[1]), 6), "lr": lr, "grad_norm": round(float(gn), 4),
                 "step_s": round(el / n_w, 4), "tokens_per_s_gpu": round(tps),
                 "mfu": round(4 * n_mfu * tps / PEAK_BF16, 4) if cuda else None,
                 "mfu_model": round(((4 * n_lin + 6 * n_lora) * tok_w + 4 * n_head * lab_w) / world / max(el, 1e-9)
                                    / PEAK_BF16, 4) if cuda else None,
                 "imbalance": round(lay.imbalance, 4), "micro_per_rank": len(lay.micro), "ckpt_share": round(ck, 4),
                 "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None,
                 "win_steps": n_w, "win_s": round(el, 4), "win_tokens": tok_w, "win_labels": lab_w, "win_rows": rows_w,
                 "step_losses": [round(s / max(1.0, c), 6) for s, c in hl], "t_end": time.time()})
            tok_w = lab_w = n_w = rows_w = 0
            eng.ckpt_layers = eng.ckpt_total = 0
            hist.clear()
            t_w = time.time()
            if flag.item() > 0 and step < end:
                log({"event": "stop_signal", "step": step, "signal": stop.signum})
                stopped = True
                break
        if val and args.eval_every and step % args.eval_every == 0 and step < end:
            do_val(step)
            t_w = time.time()
        if args.save_every and step % args.save_every == 0 and step < total:
            save(f"step-{step:06d}", step)
            t_w = time.time()
    run_s = time.time() - t_run
    if val and not stopped:
        do_val(step)
    save("final" if step >= total else f"step-{step:06d}", step)
    log({"event": "summary", "steps": step - args.start_step, "run_s": round(run_s, 2), "stopped": stopped,
         "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None,
         "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2**30, 2) if cuda else None,
         "host_offload_peak_gb": round(offload.peak_host / 2**30, 2) if offload is not None else None})
    if wb is not None:
        wb.finish()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
