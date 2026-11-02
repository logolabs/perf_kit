"""Figures for README / WRITEUP, generated from results/summary.json (bench/analysis/aggregate.py):

  figures/throughput.png   steady-state tokens/s/GPU, best tuned config per framework, 1 GPU and 4 GPUs
                           (bar = mean over repeats, ticks = each run)
  figures/loss_parity.png  training loss per step on identical step sets (1 GPU) and its difference to perf_kit
  figures/ablation.png     perf_kit with one piece removed at a time, step time vs the full engine (two repeats)
  figures/micro_sweep.png  throughput and recompute share vs micro-batch token budget, measured vs old estimate

    python plots.py --summary results/summary.json --out figures
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette, fixed order, never cycled (light mode)
COLORS = {"pk": "#2a78d6", "hf": "#eb6834", "hfx": "#1baf7a", "swift": "#eda100", "unsloth": "#e87ba4"}
NAMES = {"pk": "perf_kit", "hf": "transformers + PEFT (SDPA)", "hfx": "transformers + PEFT + FA2 + Liger",
         "swift": "ms-swift 4.5.3 (padding-free, FA2, Liger)", "unsloth": "Unsloth 2026.10.11"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, grid="x"):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cfg_label(cfg: str) -> str:
    parts = cfg.split("_")
    fw = parts[0]
    if fw == "pk":
        return f"{parts[1]}-token micro-batches"
    rest = f"{parts[1]} rows/micro-batch" if len(parts) > 1 else ""
    return rest + (", no grad. ckpt" if "nockpt" in cfg else ", grad. ckpt")


def fig_throughput(table, runs, out: Path, model="9B"):
    rows_by_g = defaultdict(list)
    for t in table:
        if t["model"] != model:
            continue
        fw = t["config"].split("_")[0]
        rows_by_g[t["gpus"]].append((fw, t))
    if not rows_by_g:
        return
    layouts = sorted(rows_by_g)
    fig, axes = plt.subplots(1, len(layouts), figsize=(6.4 * len(layouts), 3.9), facecolor=SURF)
    axes = [axes] if len(layouts) == 1 else list(axes)
    xmax = max(t["tokens_per_s_gpu_max"] for g in layouts for _, t in rows_by_g[g]) * 1.32
    for ax, g in zip(axes, layouts):
        style(ax)
        best = {}
        for fw, t in rows_by_g[g]:                   # best config per framework; perf_kit warm and cold apart
            key = fw if fw != "pk" else f"pk_{t['triton_cache']}"
            if key not in best or t["tokens_per_s_gpu_mean"] > best[key]["tokens_per_s_gpu_mean"]:
                best[key] = t
        order = [k for k in ("pk_warm", "pk_cold", "swift", "unsloth", "hfx", "hf") if k in best]
        for i, k in enumerate(order):
            t = best[k]
            fw = k.split("_")[0]
            per = [r["tokens_per_s_gpu"] for r in runs if f"{r['run']}/{r['arm']}" in t["runs"]]
            cold = k == "pk_cold"
            ax.barh(i, t["tokens_per_s_gpu_mean"], height=0.6, color=SURF if cold else COLORS[fw],
                    edgecolor=COLORS[fw], linewidth=2 if cold else 0, hatch="////" if cold else None)
            ax.plot(per, [i] * len(per), "|", color=INK, markersize=15, markeredgewidth=1.3)
            lab = f"{t['tokens_per_s_gpu_mean']:,.0f}  (n={t['n']})"
            ax.text(t["tokens_per_s_gpu_max"] + xmax * 0.015, i, lab, va="center", fontsize=8.5, color=INK)
        ylabels = []
        for k in order:
            fw = k.split("_")[0]
            nm = NAMES[fw] + (" - warm kernel cache" if k == "pk_warm" else
                              " - first process on node (JIT)" if k == "pk_cold" else "")
            ylabels.append(f"{nm}\n{cfg_label(best[k]['config'])}")
        ax.set_yticks(range(len(order)), ylabels, fontsize=8, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(f"{g} GH200{'s (one node)' if g > 1 else ''}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("tokens / s / GPU, steps 41-60 (real tokens, padding excluded)", fontsize=8.5, color=INK2)
    fig.suptitle(f"Qwen3.5-{model}-Base LoRA r16, identical 64-row steps, each framework at its best tuned setting",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_loss(runs, out: Path, job_prefix="main1"):
    sel = [r for r in runs if r["run"].startswith(job_prefix) and r["gpus"] == 1]
    if not sel:
        return
    by = {r["config"].split("_")[0]: r for r in sel}
    pk = by.get("pk")
    if not pk or not pk.get("step_losses"):
        return
    ps = {int(k): v for k, v in pk["step_losses"].items()}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.6), facecolor=SURF, sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1]})
    for ax in (a1, a2):
        style(ax, "y")
    for fw in ("pk", "hf", "hfx", "unsloth"):
        r = by.get(fw)
        if not r or not r.get("step_losses"):
            continue
        st = {int(k): v for k, v in r["step_losses"].items()}
        xs = sorted(st)
        a1.plot(xs, [st[x] for x in xs], color=COLORS[fw], linewidth=2.2 if fw == "pk" else 1.3,
                label=f"{NAMES[fw]} ({cfg_label(r['config'])})", zorder=3 if fw == "pk" else 2)
        if fw != "pk":
            a2.plot(xs, [st[x] - ps[x] for x in xs if x in ps], color=COLORS[fw], linewidth=1.3)
    sw = by.get("swift")
    if sw and sw.get("window_losses"):
        xs = sorted(int(k) for k in sw["window_losses"])
        a1.plot(xs, [sw["window_losses"][k] if k in sw["window_losses"] else sw["window_losses"][str(k)] for k in xs],
                "o", color=COLORS["swift"], markersize=8, markeredgecolor=SURF, markeredgewidth=2,
                label=NAMES["swift"] + " (its 10-step means; it also trains one extra newline token per row)")
    a2.axhline(0, color=INK2, linewidth=1)
    a1.set_ylabel("loss (mean CE per target)", fontsize=9, color=INK2)
    a2.set_ylabel("difference to perf_kit", fontsize=9, color=INK2)
    a2.set_xlabel("optimizer step (same 64 rows per step for every framework)", fontsize=9, color=INK2)
    a1.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1)
    fig.suptitle("Same math: per-step training loss on identical batches, 1 GH200", fontsize=10, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(out / "loss_parity.png", dpi=160, facecolor=SURF)
    plt.close(fig)


ABL_NAMES = {"no_fla_conv": "PyTorch per-row conv instead of fla causal conv", "no_liger": "PyTorch RMSNorm / add+norm instead of Liger",
             "no_flash_attn": "eager per-row attention instead of flash-attn varlen",
             "no_swiglu_kernel": "PyTorch SwiGLU instead of the fused kernel",
             "no_fla_gnorm": "PyTorch gated RMSNorm instead of fla's", "ckpt_all": "recompute every layer (standard)",
             "no_fused_grad_acc": "bf16 LoRA grads + add instead of fp32 addmm",
             "fa3": "flash-attn 3 instead of 2", "flashqla": "FlashQLA GDN instead of fla",
             "no_fused_gate": "GDN gate in PyTorch instead of inside fla's kernel"}


def fig_ablation(abl, out: Path, ref="mt8192"):
    ok = [a for a in abl if a.get("median_step_s") and not a.get("first_in_run")]
    by_run = defaultdict(dict)
    for a in ok:
        by_run[a["run"]].setdefault(a["config"], []).append(a)
    deltas = defaultdict(list)
    for run, cfgs in by_run.items():
        refs = cfgs.get(ref)
        if not refs:
            continue
        r0 = statistics.mean(x["median_step_s"] for x in refs)
        for c, xs in cfgs.items():
            if c in ABL_NAMES:
                deltas[c].append(100 * (statistics.mean(x["median_step_s"] for x in xs) / r0 - 1))
    if not deltas:
        return
    rows = sorted(deltas, key=lambda c: statistics.mean(deltas[c]))
    fig, ax = plt.subplots(figsize=(9.2, 0.44 * len(rows) + 1.4), facecolor=SURF)
    style(ax)
    for i, c in enumerate(rows):
        m = statistics.mean(deltas[c])
        ax.barh(i, m, height=0.58, color=COLORS["hf"] if m > 0 else COLORS["pk"], edgecolor=SURF, linewidth=2)
        ax.plot(deltas[c], [i] * len(deltas[c]), "|", color=INK, markersize=13, markeredgewidth=1.2)
        ax.text(max(0.0, max(deltas[c])) + 1.0, i,
                f"{m:+.1f}%" + (f"  (n={len(deltas[c])})" if len(deltas[c]) > 1 else ""), va="center",
                ha="left", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)), [ABL_NAMES[c] for c in rows], fontsize=8.5, color=INK)
    lo = min(min(v) for v in deltas.values())
    hi = max(max(v) for v in deltas.values())
    ax.set_xlim(min(-8, lo - 2), hi + 14)
    ax.set_xlabel("step time change vs full perf_kit (1 GH200, 8,192-token micro-batches; + = slower)",
                  fontsize=8.5, color=INK2)
    fig.suptitle("Remove one piece at a time (per-step losses unchanged in every case)", fontsize=10, color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_sweep(abl, out: Path):
    def pts(prefix):
        d = defaultdict(list)
        for a in abl:
            if a.get("median_step_s") and a["config"].startswith(prefix) and not a.get("first_in_run"):
                d[a["micro_tokens"]].append(a)
        return sorted(d.items())
    new, old = pts("mt"), pts("old_planner")
    if not new:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.5), facecolor=SURF)
    for ax in (a1, a2):
        style(ax, "y")
    nx = [k for k, _ in new]
    a1.plot(nx, [statistics.mean(x["tokens_per_s_gpu"] for x in v) for _, v in new], "-o", color=COLORS["pk"], linewidth=2,
            markersize=8, markeredgecolor=SURF, markeredgewidth=2, label="activation size measured at start-up")
    a2.plot(nx, [100 * v[-1]["ckpt_share"] for _, v in new], "-o", color=COLORS["pk"], linewidth=2, markersize=8,
            markeredgecolor=SURF, markeredgewidth=2)
    if old:
        ox = [k for k, _ in old]
        a1.plot(ox, [statistics.mean(x["tokens_per_s_gpu"] for x in v) for _, v in old], "s", color=COLORS["hf"], markersize=9,
                markeredgecolor=SURF, markeredgewidth=2, label="old fixed estimate (0.25 MiB/token/layer)")
        a2.plot(ox, [100 * v[-1]["ckpt_share"] for _, v in old], "s", color=COLORS["hf"], markersize=9,
                markeredgecolor=SURF, markeredgewidth=2)
    a1.set_xlabel("micro-batch token budget", fontsize=9, color=INK2)
    a2.set_xlabel("micro-batch token budget", fontsize=9, color=INK2)
    a1.set_ylabel("tokens / s / GPU (median step)", fontsize=9, color=INK2)
    a2.set_ylabel("layers recomputed (%)", fontsize=9, color=INK2)
    a1.set_ylim(0, None)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    a1.set_title("Throughput vs micro-batch size, 1 GH200", fontsize=10, color=INK, loc="left")
    a2.set_title("Recompute the planner chose", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "micro_sweep.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args(argv)
    s = json.loads(Path(a.summary).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_throughput(s["table"], s["runs"], out)
    fig_loss(s["runs"], out)
    fig_ablation(s["ablations"], out)
    fig_sweep(s["ablations"], out)
    print(sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
