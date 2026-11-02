"""Figures for README / WRITEUP, generated from results/summary.json (bench/analysis/aggregate.py):

  figures/throughput.png   tokens/s/GPU per framework, 1 GPU and 1 node (bars = mean over repeats, ticks = each run)
  figures/loss_parity.png  training loss per step on identical step sets, 1 GPU
  figures/ablation.png     perf_kit with one trick removed at a time: throughput relative to the full engine
  figures/micro_sweep.png  throughput and peak memory vs micro-batch token budget, measured vs old activation estimate

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

# reference categorical palette (fixed order, never cycled), light mode
COLORS = {"pk": "#2a78d6", "hf": "#eb6834", "hfx": "#1baf7a", "swift": "#eda100"}
NAMES = {"pk": "perf_kit", "hf": "transformers + PEFT", "hfx": "transformers + PEFT\n+ FA2 + Liger",
         "swift": "ms-swift 4.5.3\n(padding-free, FA2, Liger)"}
ORDER = ["pk", "swift", "hfx", "hf"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_throughput(runs, out: Path):
    groups = defaultdict(list)
    for r in runs:
        if "tokens_per_s_gpu" in r and r["arm"] in COLORS:
            groups[(r["gpus"], r["arm"])].append(r["tokens_per_s_gpu"])
    layouts = sorted({g for g, _ in groups})
    if not layouts:
        return
    fig, axes = plt.subplots(1, len(layouts), figsize=(5.2 * len(layouts), 3.4), sharex=True, facecolor=SURF)
    axes = [axes] if len(layouts) == 1 else list(axes)
    xmax = max(max(v) for v in groups.values()) * 1.22
    for ax, g in zip(axes, layouts):
        style(ax)
        arms = [a for a in ORDER if (g, a) in groups]
        for i, a in enumerate(arms):
            v = groups[(g, a)]
            m = statistics.mean(v)
            ax.barh(i, m, height=0.56, color=COLORS[a], edgecolor=SURF, linewidth=2)
            ax.plot(v, [i] * len(v), "|", color=INK, markersize=14, markeredgewidth=1.2)
            spread = f"  (n={len(v)}, {min(v):,.0f}-{max(v):,.0f})" if len(v) > 1 else "  (n=1)"
            ax.text(m + xmax * 0.01, i, f"{m:,.0f}{spread}", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(arms)), [NAMES[a] for a in arms], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(f"{g} GH200{'s' if g > 1 else ''}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training tokens / s / GPU (real tokens, no padding)", fontsize=9, color=INK2)
    fig.suptitle("Qwen3.5-9B-Base LoRA r16, identical step sets", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "throughput.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_loss(runs, out: Path):
    one = [r for r in runs if r["gpus"] == 1 and r.get("step_losses")]
    if not one:
        return
    first = {}
    for r in one:                          # first run of each arm
        first.setdefault(r["arm"], r)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for a in ORDER:
        r = first.get(a)
        if not r or a not in COLORS:
            continue
        if r["step_losses"]:
            xs = sorted(int(k) for k in r["step_losses"])
            ax.plot(xs, [r["step_losses"][k] if k in r["step_losses"] else r["step_losses"][str(k)] for k in xs],
                    color=COLORS[a], linewidth=2 if a == "pk" else 1.4, label=NAMES[a].replace("\n", " "),
                    alpha=1 if a == "pk" else 0.9)
    for a in ORDER:
        r = first.get(a)
        if r and r.get("window_losses"):
            xs = sorted(int(k) for k in r["window_losses"])
            ax.plot(xs, [r["window_losses"][k] for k in xs], "o", color=COLORS[a], markersize=8,
                    markeredgecolor=SURF, markeredgewidth=2, label=NAMES[a].replace("\n", " ") + " (10-step means)")
    ax.set_xlabel("optimizer step", fontsize=9, color=INK2)
    ax.set_ylabel("training loss (mean CE per target)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("Same math: per-step loss on identical batches, 1 GH200", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "loss_parity.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_ablation(abl, out: Path, base_cfg: str = "base"):
    ok = [a for a in abl if a.get("tokens_per_s_gpu")]
    base = next((a for a in ok if a["config"] == base_cfg), None)
    if not base:
        return
    rows = [a for a in ok if not a["config"].startswith(("mt", "old_planner")) and a["config"] != base_cfg]
    if not rows:
        return
    rows.sort(key=lambda a: a["tokens_per_s_gpu"])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2), facecolor=SURF)
    style(ax)
    for i, a in enumerate(rows):
        d = 100 * (a["tokens_per_s_gpu"] / base["tokens_per_s_gpu"] - 1)
        ax.barh(i, d, height=0.56, color=COLORS["pk"] if d >= 0 else COLORS["hf"], edgecolor=SURF, linewidth=2)
        ax.text(d + (1 if d >= 0 else -1), i, f"{d:+.1f}%", va="center", ha="left" if d >= 0 else "right",
                fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)), [a["config"] for a in rows], fontsize=9, color=INK)
    lo = min(100 * (a["tokens_per_s_gpu"] / base["tokens_per_s_gpu"] - 1) for a in rows)
    ax.set_xlim(min(-5, lo * 1.35), 15)
    ax.set_xlabel(f"tokens/s change vs full perf_kit ({base['tokens_per_s_gpu']:,.0f} tok/s, 1 GH200)",
                  fontsize=9, color=INK2)
    ax.set_title("Remove one piece at a time", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_sweep(abl, out: Path):
    new = sorted((a for a in abl if a.get("tokens_per_s_gpu") and a["config"].startswith("mt")),
                 key=lambda a: a["micro_tokens"])
    old = sorted((a for a in abl if a.get("tokens_per_s_gpu") and a["config"].startswith("old_planner")),
                 key=lambda a: a["micro_tokens"])
    if not new:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.4), facecolor=SURF)
    for ax, key, lab in ((a1, "tokens_per_s_gpu", "tokens / s / GPU"), (a2, "vram_peak_gb", "peak allocated GB")):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.plot([a["micro_tokens"] for a in new], [a[key] for a in new], "-o", color=COLORS["pk"], linewidth=2,
                markersize=8, markeredgecolor=SURF, markeredgewidth=2, label="measured activation size (new)")
        if old:
            ax.plot([a["micro_tokens"] for a in old], [a[key] for a in old], "s", color=COLORS["hf"], markersize=9,
                    markeredgecolor=SURF, markeredgewidth=2, label="fixed 0.25 MiB/token/layer (old)")
        ax.set_xlabel("micro-batch token budget", fontsize=9, color=INK2)
        ax.set_ylabel(lab, fontsize=9, color=INK2)
    for a in new:
        a1.annotate(f"recompute {100 * a['ckpt_share']:.0f}%", (a["micro_tokens"], a["tokens_per_s_gpu"]),
                    textcoords="offset points", xytext=(0, -16), ha="center", fontsize=7.5, color=INK2)
    a1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    a1.set_title("Throughput vs micro-batch size, 1 GH200", fontsize=10, color=INK, loc="left")
    a2.set_title("Peak memory", fontsize=10, color=INK, loc="left")
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
    fig_throughput(s["runs"], out)
    fig_loss(s["runs"], out)
    fig_ablation(s["ablations"], out)
    fig_sweep(s["ablations"], out)
    print(sorted(p.name for p in out.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
