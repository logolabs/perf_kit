"""perf_kit ablations on one GPU with ONE model load: every configuration trains the same sequential step sets from
the same LoRA init with a fresh AdamW, so step times compare directly and the per-step losses double as a numerical
check (an exact trick must not move them beyond bf16 noise). Each step is CUDA-synchronised (a small, uniform
overhead), so per-step times are available; the reported time is the median of the steps after `--warm`.

    python pk_ablate.py --model <hf dir> --train train.jsonl --init_adapter <dir> --out <dir> \
        --configs base mt6144 old_planner_mt8192 no_liger fa3 ...

Configurations (see CONFIGS): micro-batch token budgets, the old fixed activation estimate (0.25 MiB / token /
layer) vs the start-up measurement, full recompute, and each fast kernel forced back to its reference path.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

for _d in (Path(__file__).resolve().parent, Path(sys.argv[0]).resolve().parent):
    if (_d / "common.py").exists():
        sys.path.insert(0, str(_d))
        break
from common import JsonLog, load_rows  # noqa: E402

CONFIGS = {
    "base": {},
    **{f"mt{t}": {"micro_tokens": t} for t in (3072, 4096, 6144, 8192, 10240, 12288, 16384)},
    "old_planner_mt8192": {"micro_tokens": 8192, "act_mb": 0.25, "in_mb": 0.01},
    "old_planner_mt12288": {"micro_tokens": 12288, "act_mb": 0.25, "in_mb": 0.01},
    "ckpt_all": {"ckpt": "all"},
    "no_liger": {"disable": ["liger_rms", "liger_add_rms"]},
    "no_fla_conv": {"disable": ["fla_conv"]},
    "no_swiglu_kernel": {"disable": ["triton"]},
    "no_fused_gate": {"disable": ["gdn_fused_gate"]},
    "no_fla_gnorm": {"disable": ["fla_gnorm"]},
    "no_flash_attn": {"disable": ["flash"]},
    "no_fused_grad_acc": {"fused_acc": False},
    "fa3": {"attn": "fa3"},
    "flashqla": {"gdn": "flashqla"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--model", required=True)
    a("--train", nargs="+", required=True)
    a("--init_adapter", required=True)
    a("--out", required=True)
    a("--configs", nargs="+", default=["base"])
    a("--chat_template_kwargs", default="")
    a("--rank", type=int, default=16)
    a("--alpha", type=float, default=32)
    a("--lr", type=float, default=2e-4)
    a("--global_batch", type=int, default=64)
    a("--steps", type=int, default=16)
    a("--warm", type=int, default=6)
    a("--micro_tokens", type=int, default=8192)
    a("--storage", default="bf16", choices=("bf16", "fp8"))
    a("--max_len", type=int, default=4096)
    a("--mem_frac", type=float, default=0.92)
    a("--reserve_gb", type=float, default=6.0)
    args = ap.parse_args(argv)
    import torch
    from perf_kit import lora as LO
    from perf_kit.engine import CkptPolicy, Engine, Prefetcher, lr_cosine, measure_act_mb
    from perf_kit.kernels import ops
    from perf_kit.layout import SequentialSteps, plan_step
    from perf_kit.packing import pack
    from perf_kit.train import build_model, versions
    from perf_kit.weights import load_adapter
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; known: {sorted(CONFIGS)}")
    torch.backends.cuda.enable_cudnn_sdp(False)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    log = JsonLog(Path(args.out), 0)
    tkw = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    log({"event": "config", "harness": "pk_ablate", "args": vars(args), "versions": versions(),
         "device": torch.cuda.get_device_name(dev), "kernels": ops.available()})
    _, rows, _, _ = load_rows(args.model, args.train, tkw, args.max_len, limit=args.global_batch * args.steps)
    ns = argparse.Namespace(model=args.model, rank=args.rank, alpha=args.alpha, storage=args.storage, impl="auto",
                            targets=None)
    t0 = time.time()
    model = build_model(ns, dev, torch.bfloat16)
    tensors, _ = load_adapter(args.init_adapter)
    model.lora.load_peft(tensors)
    init = model.lora.master.detach().clone()
    lengths = [rows.length(i) for i in range(len(rows))]
    n_labels = [rows.n_labels(i) for i in range(len(rows))]
    total = torch.cuda.get_device_properties(dev).total_memory / 2**30
    static = torch.cuda.memory_allocated(dev) / 2**30
    budget = max(0.0, args.mem_frac * total - static - 2 * model.lora.master.numel() * 4 / 2**30 - args.reserve_gb)
    probe, n_tok = [], 0
    for i in range(len(rows)):
        if probe and n_tok + lengths[i] > 2048:
            break
        probe.append(rows[i])
        n_tok += lengths[i]
    calib = measure_act_mb(model, pack(probe).to(dev))
    log({"event": "model", "secs": round(time.time() - t0, 1), "static_gb": round(static, 2),
         "act_budget_gb": round(budget, 2), "act_calibration": calib, "lora_params": model.lora.numel,
         "frozen_linear_params": model.frozen_linear_params(), "head_params": model.head_params(),
         "n_layers": len(model.layers)})
    steps = SequentialSteps(len(rows), args.global_batch)
    pre = Prefetcher(lambda i: rows[i], dev)
    defaults = (ops.DISABLED, ops.ATTN_BACKEND, ops.GDN_BACKEND)
    for name in args.configs:
        c = CONFIGS[name]
        ops.DISABLED = frozenset(c.get("disable", ()))
        ops.ATTN_BACKEND = c.get("attn", defaults[1])
        ops.GDN_BACKEND = c.get("gdn", defaults[2])
        LO._FUSED_ACC["ok"] = None if c.get("fused_acc", True) else False
        need = {"fa3": "fa3", "flashqla": "flashqla"}
        miss = [lib for key, lib in need.items() if (c.get("attn") == key or c.get("gdn") == key)
                and ops._lib(lib) is None]
        if miss:
            log({"event": "ablation", "config": name, "skipped": f"library not importable: {miss}"})
            continue
        pol = CkptPolicy(len(model.layers), budget, mode=c.get("ckpt", "auto"))
        if "act_mb" in c:
            pol.act_mb, pol.in_mb = c["act_mb"], c["in_mb"]
        else:
            pol.calibrate(calib["act_mb"], calib["in_mb"])
        mt = c.get("micro_tokens", args.micro_tokens)
        with torch.no_grad():
            model.lora.master.copy_(init)
            model.lora.push()
        eng = Engine(model, lr=args.lr, ckpt=pol, segs=(2,))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        times, losses, toks, labs, err = [], [], [], [], None
        try:
            for s in range(args.steps):
                lay = plan_step(steps.step(s), lengths, n_labels, 0, 1, mt)
                t = time.time()
                eng.train_step(lay, pre.run(lay.micro), lr_cosine(s, args.steps, args.lr, 2, 0.1))
                v = eng.acc_step[:2].tolist()                 # syncs: per-step timing
                times.append(time.time() - t)
                losses.append(v[0] / max(1.0, v[1]))
                toks.append(lay.tokens_total)
                labs.append(lay.label_total)
        except torch.OutOfMemoryError as ex:
            err = f"OOM: {str(ex)[:160]}"
        meas = times[args.warm:]
        rec = {"event": "ablation", "config": name, "settings": c, "micro_tokens": mt, "error": err,
               "ckpt_act_mb": round(pol.act_mb, 4), "ckpt_share": round(eng.ckpt_layers / max(1, eng.ckpt_total), 4),
               "steps_measured": len(meas), "median_step_s": round(statistics.median(meas), 4) if meas else None,
               "mean_step_s": round(statistics.mean(meas), 4) if meas else None,
               "tokens_measured": sum(toks[args.warm:]), "labels_measured": sum(labs[args.warm:]),
               "seconds_measured": round(sum(meas), 4),
               "vram_peak_gb": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2),
               "step_s_all": [round(x, 4) for x in times], "step_losses": [round(x, 6) for x in losses],
               "kernels": ops.available()}
        log(rec)
        del eng
    ops.DISABLED, ops.ATTN_BACKEND, ops.GDN_BACKEND = defaults
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
