"""Draw assets/masthead.png for the README from results/summary.json (the same numbers as the results table).

    python bench/analysis/make_masthead.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BG, INK, MUTED, BLUE, LINE = "#FCFCFB", "#1D1F24", "#5F6670", "#2A78D6", "#E2E4E8"


def tps(rows, gpus, config, cache="n/a"):
    for r in rows:
        if r["model"] == "9B" and r["gpus"] == gpus and r["config"] == config and r["triton_cache"] == cache:
            return r["tokens_per_s_gpu_mean"], r["mfu_mean"], r["mfu_clock_adjusted_mean"]
    raise KeyError((gpus, config, cache))


def main():
    return draw(json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))["table"])


def draw(rows):
    pk, mfu, mfu_adj = tps(rows, 1, "pk_12288", "warm")
    tiles = [(f"{pk / tps(rows, 1, cfg)[0]:.2f}x", label) for cfg, label in
             (("swift_8_nockpt", "faster than ms-swift"), ("unsloth_8", "faster than Unsloth"),
              ("hf_8", "faster than transformers + PEFT"))]
    tiles.append((f"{100 * mfu:.1f}%", f"MFU ({100 * mfu_adj:.1f}% clock-adjusted)"))
    fig = plt.figure(figsize=(20.48, 5.6), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 560)
    ax.axis("off")
    ax.text(80, 420, "perf_kit", fontsize=64, fontweight="bold", color=INK, va="center")
    ax.text(84, 330, "Fast, exact LoRA fine-tuning for Qwen3.5-style hybrid LLMs on NVIDIA GH200",
            fontsize=21, color=MUTED, va="center")
    x0, w, gap = 80, 452, 26
    for i, (big, small) in enumerate(tiles):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 70), w, 170, boxstyle="round,pad=0,rounding_size=14",
                                    facecolor="#FFFFFF", edgecolor=LINE, linewidth=1.5))
        ax.text(x + 28, 182, big, fontsize=44, fontweight="bold", color=BLUE if i < 3 else INK, va="center")
        ax.text(x + 30, 112, small, fontsize=16, color=MUTED, va="center")
    ax.text(84, 30, "Qwen3.5-9B-Base, LoRA r16, one GH200, identical 64-row steps, each framework at its best setting; "
            "per-step losses match transformers + PEFT to < 6e-4", fontsize=13, color=MUTED, va="center")
    out = ROOT / "assets" / "masthead.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, facecolor=BG)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
