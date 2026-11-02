"""Baseline: LoRA SFT with plain transformers + PEFT (the way most people run it), on exactly the workload
perf_kit.train runs in the benchmark, so the comparison isolates the execution engine:

  * same token ids and labels (perf_kit.data encodes the rows), same step sets (sequential: step s = rows
    [s*gb, (s+1)*gb)), same init adapter (make_init_adapter.py), same LR schedule (perf_kit.engine.lr_cosine),
    same AdamW (betas 0.9/0.99, eps 1e-8, no weight decay, fused), same grad clipping (1.0);
  * the loss of a step is sum(CE) / (trained targets of the whole step, all ranks) -- what perf_kit and recent
    transformers.Trainer (num_items_in_batch, average_tokens_across_devices) compute;
  * standard execution: bf16 base weights, fp32 LoRA (PEFT's autocast_adapter_dtype) under torch.autocast(bf16),
    right-padded micro-batches of --micro_bs rows (rows of a step sorted by length first = group_by_length), DDP with
    no_sync during accumulation, optional gradient checkpointing (non-reentrant), attention through
    --attn {sdpa, flash_attention_2, eager}, optional Liger kernels (RMSNorm, SwiGLU, fused linear + CE).

Padding-free (packed) batches are NOT used: transformers' Qwen3.5 short convolution falls back to a PyTorch conv
that ignores packed-row boundaries unless the causal-conv1d package is installed (it is not on aarch64 here), so
packing would change the math.

    python hf_peft_train.py --model <hf dir> --train train.jsonl --val val.jsonl --out <dir> --init_adapter <dir> \
        --global_batch 64 --micro_bs 8 --max_steps 60 --grad_ckpt [--attn flash_attention_2 --liger]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path

# perun (energy monitor) execs scripts in-process with __file__ set to its app name: find our directory either way
for _d in (Path(__file__).resolve().parent, Path(sys.argv[0]).resolve().parent):
    if (_d / "common.py").exists():
        sys.path.insert(0, str(_d))
        break
from common import TARGETS, JsonLog, dist_env, load_rows, padded_batch  # noqa: E402

PEAK_BF16 = 989e12


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--model", required=True)
    a("--train", nargs="+", required=True)
    a("--val", default="")
    a("--out", required=True)
    a("--init_adapter", required=True)
    a("--chat_template_kwargs", default="")
    a("--rank", type=int, default=16)
    a("--alpha", type=float, default=32)
    a("--lr", type=float, default=2e-4)
    a("--warmup", type=int, default=6)
    a("--min_lr_ratio", type=float, default=0.1)
    a("--max_steps", type=int, default=60)
    a("--global_batch", type=int, default=64)
    a("--micro_bs", type=int, default=8)
    a("--max_len", type=int, default=4096)
    a("--val_rows", type=int, default=256)
    a("--log_every", type=int, default=10)
    a("--attn", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    a("--liger", action="store_true")
    a("--grad_ckpt", action="store_true")
    a("--unsloth", action="store_true", help="load + LoRA-wrap the model with Unsloth (its patched kernels); the "
      "training loop, data, init and optimizer stay the same")
    a("--device", default="cuda", choices=("cuda", "cpu"))
    a("--token_cache", default="")
    a("--eval_only", action="store_true", help="only the val loss of --init_adapter (e.g. another trainer's result)")
    return ap.parse_args(argv)


def kernel_report() -> dict:
    import importlib.util as u
    return {k: u.find_spec(k) is not None for k in ("fla", "causal_conv1d", "flash_attn", "liger_kernel")}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.unsloth:
        import unsloth  # noqa: F401  (must be imported before transformers to patch it)
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file
    from perf_kit.engine import lr_cosine
    from perf_kit.layout import SequentialSteps
    rank, world, local = dist_env()
    cuda = args.device == "cuda"
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    if cuda:
        torch.cuda.set_device(local)
        torch.backends.cuda.enable_cudnn_sdp(False)     # cuDNN SDPA returned NaNs on GH200 (torch 2.13); same as perf_kit
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    out = Path(args.out)
    log = JsonLog(out, rank)
    tkw = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    import transformers
    import peft
    log({"event": "config", "harness": "hf_peft", "args": vars(args), "world": world,
         "versions": {"python": platform.python_version(), "torch": torch.__version__,
                      "transformers": transformers.__version__, "peft": peft.__version__},
         "kernels": kernel_report(), "device": torch.cuda.get_device_name(dev) if cuda else "cpu"})

    t0 = time.time()
    tok, rows, dropped, src = load_rows(args.model, args.train, tkw, args.max_len, 0,
                                        args.token_cache if rank == 0 else "")
    if world > 1:
        dist.barrier()
        if rank != 0 and args.token_cache:
            tok, rows, dropped, src = load_rows(args.model, args.train, tkw, args.max_len, 0, args.token_cache)
    val = None
    if args.val:
        _, val, _, _ = load_rows(args.model, [args.val], tkw, args.max_len, args.val_rows)
    log({"event": "data", "rows": len(rows), "dropped_long": dropped, "tokens": int(rows.off[-1]),
         "val": len(val) if val is not None else 0, "source": src, "secs": round(time.time() - t0, 1)})

    t0 = time.time()
    if args.liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
        apply_liger_kernel_to_qwen3_5(rms_norm=True, swiglu=True, fused_linear_cross_entropy=True)
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16 if cuda else torch.float32
    if args.unsloth:
        from unsloth import FastLanguageModel
        model, _ = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_len, dtype=dtype,
                                                     load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
                                                     local_files_only=True)
        model = FastLanguageModel.get_peft_model(model, r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0,
                                                 bias="none", target_modules=list(TARGETS), random_state=0,
                                                 use_gradient_checkpointing="unsloth" if args.grad_ckpt else False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation=args.attn)
        model.config.use_cache = False
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.use_cache = False
        model.to(dev)
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lcfg = peft.LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                               target_modules=list(TARGETS), task_type="CAUSAL_LM")
        model = peft.get_peft_model(model, lcfg)
    # load the shared init adapter, matched by the "layers.N.<module>.lora_X.weight" suffix: the checkpoint (and
    # perf_kit) name the decoder model.language_model.layers.*, transformers' text-only class model.layers.*
    init = load_file(str(Path(args.init_adapter) / "adapter_model.safetensors"))
    by_suffix = {k[k.index("layers."):]: v for k, v in init.items()}
    if len(by_suffix) != len(init):
        raise SystemExit("init adapter: ambiguous layer suffixes")
    n_set = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                k = n.replace(".default", "")
                k = k[k.index("layers."):]
                if k not in by_suffix:
                    raise SystemExit(f"init adapter has no ...{k}")
                p.copy_(by_suffix[k].to(p.dtype))
                n_set += 1
    if n_set != len(init):
        raise SystemExit(f"init adapter: {len(init)} tensors, model has {n_set} LoRA tensors")
    a_sum = float(sum(p.detach().double().abs().sum() for n, p in model.named_parameters() if "lora_A" in n))
    params = [p for p in model.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in params)
    lora_dtypes = sorted({str(p.dtype) for p in params})
    ddp = model
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None,
                                                        gradient_as_bucket_view=True)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, fused=cuda)
    log({"event": "model", "class": type(model.get_base_model()).__name__, "lora_params": n_lora,
         "unsloth": args.unsloth,
         "lora_dtypes": lora_dtypes, "lora_A_abs_sum": a_sum, "attn": args.attn, "liger": args.liger, "grad_ckpt": args.grad_ckpt,
         "static_gb": round(torch.cuda.memory_allocated(dev) / 2**30, 2) if cuda else None,
         "secs": round(time.time() - t0, 1)})
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    lengths = [rows.length(i) for i in range(len(rows))]
    n_labels = [rows.n_labels(i) for i in range(len(rows))]
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cuda else contextlib.nullcontext

    def micro_loss(batch, n_targets):
        """sum of CE over the micro-batch's targets (the model returns the mean)."""
        with autocast():
            out_ = ddp(**batch)
        return out_.loss.float() * n_targets

    @torch.no_grad()
    def run_val():
        ddp.eval()
        idx = list(range(rank, len(val), world))
        idx.sort(key=lambda i: -val.length(i))
        tot = torch.zeros(2, dtype=torch.float64, device=dev)
        for j in range(0, len(idx), args.micro_bs):
            b, _, _, nl = padded_batch(val, idx[j:j + args.micro_bs], pad, dev)
            if nl:
                tot[0] += micro_loss(b, nl).double()
                tot[1] += nl
        if world > 1:
            dist.all_reduce(tot)
        ddp.train()
        v = tot.tolist()
        return round(v[0] / max(1.0, v[1]), 6), int(v[1])

    if val is not None:
        vl, vn = run_val()
        log({"event": "val", "step": 0, "val_loss": vl, "val_targets": vn})
    if args.eval_only:
        return 0
    steps = SequentialSteps(len(rows), args.global_batch)
    ddp.train()
    hist, step = [], 0
    tok_w = lab_w = rows_w = pad_w = n_w = 0
    if cuda:
        torch.cuda.synchronize(dev)
    t_w = t_run = time.time()
    while step < args.max_steps:
        rows_s = steps.step(step)
        label_total = sum(n_labels[i] for i in rows_s)
        mine = sorted(rows_s[rank::world], key=lambda i: -lengths[i])        # DistributedSampler-like + length groups
        micros = [mine[j:j + args.micro_bs] for j in range(0, len(mine), args.micro_bs)]
        lr = lr_cosine(step, args.max_steps, args.lr, args.warmup, args.min_lr_ratio)
        for g in opt.param_groups:
            g["lr"] = lr
        acc = torch.zeros(2, dtype=torch.float64, device=dev)
        for j, mb in enumerate(micros):
            b, real, padded, nl = padded_batch(rows, mb, pad, dev)
            sync_ctx = ddp.no_sync() if (world > 1 and j < len(micros) - 1) else contextlib.nullcontext()
            with sync_ctx:
                s = micro_loss(b, nl)
                # DDP averages gradients over ranks, so each rank scales by world: the total is sum(CE) / label_total
                (s * (world / label_total)).backward()
            acc[0] += s.detach().double()
            acc[1] += nl
            pad_w += padded
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append(acc)
        step += 1
        n_w += 1
        tok_w += sum(lengths[i] for i in rows_s)
        lab_w += label_total
        rows_w += len(rows_s)
        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            if cuda:
                torch.cuda.synchronize(dev)
            el = time.time() - t_w
            H = torch.stack(hist)
            pw = torch.tensor([float(pad_w)], dtype=torch.float64, device=dev)
            if world > 1:
                dist.all_reduce(H)
                dist.all_reduce(pw)
            hl = H.tolist()
            log({"step": step, "loss": round(sum(x[0] for x in hl) / max(1.0, sum(x[1] for x in hl)), 6), "lr": lr,
                 "step_s": round(el / n_w, 4), "tokens_per_s_gpu": round(tok_w / world / max(el, 1e-9)),
                 "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None,
                 "win_steps": n_w, "win_s": round(el, 4), "win_tokens": tok_w, "win_labels": lab_w,
                 "win_rows": rows_w, "win_padded_tokens": int(pw.item()),
                 "step_losses": [round(x[0] / max(1.0, x[1]), 6) for x in hl], "t_end": time.time()})
            hist, tok_w, lab_w, rows_w, pad_w, n_w = [], 0, 0, 0, 0, 0
            t_w = time.time()
    run_s = time.time() - t_run
    if val is not None:
        vl, vn = run_val()
        log({"event": "val", "step": step, "val_loss": vl, "val_targets": vn})
    log({"event": "summary", "steps": step, "run_s": round(run_s, 2),
         "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2) if cuda else None,
         "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2**30, 2) if cuda else None})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
