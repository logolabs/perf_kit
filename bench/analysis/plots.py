"""Figures for README / WRITEUP / report, generated from results/summary.json (bench/analysis/aggregate.py), in the
LogoLabs house style (bench/analysis/house.py: HTML + inline SVG, rendered by headless Chromium):

  figures/throughput.png      steady-state tokens/s/GPU, best tuned config per framework, 1 GPU and 4 GPUs
                              (bar = mean over repeats, ticks = each run)
  figures/loss_parity.png     training loss per step on identical step sets (1 GPU) and its difference to perf_kit
  figures/ablation.png        perf_kit with one piece removed at a time, step time vs the full engine (two repeats)
  figures/micro_sweep.png     throughput and recompute share vs micro-batch token budget, measured vs old estimate
  assets/headline_numbers.png the README's numbers strip under the masthead

The figures are also copied to report/img/ for report/index.html, and each page's HTML is kept in
bench/analysis/html/.

    python bench/analysis/plots.py [--summary results/summary.json --out figures]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house import GREYS, GRID, INK, PAPER, RED, esc, lockup, page, render  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"pk": "perf_kit", "hf": "transformers + PEFT (SDPA)", "hfx": "transformers + PEFT + FA2 + Liger",
         "swift": "ms-swift 4.5.3 (padding-free, FA2, Liger)", "unsloth": "Unsloth 2026.10.11"}
# perf_kit is the red; the baselines are black to light grey, darkest = fastest baseline
COLORS = {"pk": RED, "swift": GREYS[0], "unsloth": GREYS[1], "hfx": GREYS[2], "hf": GREYS[3]}
SOURCE = "Source: results/summary.json, computed from results/raw by bench/analysis/aggregate.py"
MINUS = "−"


def signed(v: float, fmt: str) -> str:
    return format(v, fmt).replace("-", MINUS)


def cfg_label(cfg: str) -> str:
    parts = cfg.split("_")
    if parts[0] == "pk":
        return f"{int(parts[1]):,}-token micro-batches"
    rest = f"{parts[1]} rows/micro-batch" if len(parts) > 1 else ""
    return rest + (", no grad. ckpt" if "nockpt" in cfg else ", grad. ckpt")


class Plot:
    """A data rectangle W x H in CSS px, ink axes on the left and bottom, mono tick labels outside it."""

    def __init__(self, W, H, x0, x1, y0, y1):
        self.W, self.H, self.x0, self.x1, self.y0, self.y1 = W, H, x0, x1, y0, y1

    def X(self, v):
        return (v - self.x0) / (self.x1 - self.x0) * self.W

    def Y(self, v):
        return self.H - (v - self.y0) / (self.y1 - self.y0) * self.H

    def html(self, inner: str, xticks, yticks, fx, fy, xcap: str, ycap: str, overlay: str = "",
             margin: str = "0 0 48px 64px", xlabels: bool = True) -> str:
        W, H = self.W, self.H
        g = "".join(f"<line x1='0' x2='{W}' y1='{self.Y(v):.1f}' y2='{self.Y(v):.1f}' stroke='{GRID}'/>" for v in yticks)
        g += "".join(f"<line x1='{self.X(v):.1f}' x2='{self.X(v):.1f}' y1='0' y2='{H}' stroke='{GRID}'/>" for v in xticks)
        svg = f"<svg width='{W}' height='{H}' style='position:absolute;left:0;top:0;overflow:visible'>{g}{inner}</svg>"
        xt = "".join(f"<span style='position:absolute;left:{self.X(v) - 40:.0f}px;width:80px;text-align:center;"
                     f"top:{H + 9}px'>{fx(v)}</span>" for v in xticks) if xlabels else ""
        yt = "".join(f"<span style='position:absolute;right:calc(100% + 10px);top:{self.Y(v) - 8:.0f}px;white-space:nowrap'>"
                     f"{fy(v)}</span>" for v in yticks)
        cap = (f"<div class='axcap' style='position:absolute;left:0;top:{H + 30}px;white-space:nowrap'>{xcap}</div>"
               if xcap else "")
        return (f"<div style='position:relative;width:{W}px;height:{H}px;margin:{margin};border-left:1.5px solid var(--ink);"
                f"border-bottom:1.5px solid var(--ink)'>{svg}{overlay}<div class='axis'>{xt}{yt}</div>{cap}"
                f"<div class='axcap' style='position:absolute;left:12px;top:10px;white-space:nowrap;background:var(--paper);"
                f"padding:0 4px'>{ycap}</div></div>")


def poly(pts, color, width, dash=None, plot=None) -> str:
    d = " ".join(f"{plot.X(x):.1f},{plot.Y(y):.1f}" for x, y in pts)
    da = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<polyline fill='none' stroke='{color}' stroke-width='{width}' stroke-linejoin='round'{da} "
            f"points='{d}'/>")


def swatch(color, width=3, dash=None, square=False, hatch=False) -> str:
    if square:
        return f"<span style='display:inline-block;width:12px;height:12px;background:{color}'></span>"
    if hatch:
        return (f"<span style='display:inline-block;width:22px;height:12px;border:1.5px solid {color};"
                f"background:repeating-linear-gradient(135deg,transparent 0 3px,{color} 3px 4.5px)'></span>")
    da = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<svg width='26' height='10'><line x1='0' x2='26' y1='5' y2='5' stroke='{color}' stroke-width='{width}'{da}/>"
            f"</svg>")


# ----------------------------------------------------------------------------------------------- throughput
def best_per_framework(table, model="9B"):
    rows_by_g = defaultdict(list)
    for t in table:
        if t["model"] == model:
            rows_by_g[t["gpus"]].append((t["config"].split("_")[0], t))
    best = {}
    for g, rows in rows_by_g.items():
        b = {}
        for fw, t in rows:                       # best config per framework; perf_kit warm and cold apart
            key = fw if fw != "pk" else f"pk_{t['triton_cache']}"
            if key not in b or t["tokens_per_s_gpu_mean"] > b[key]["tokens_per_s_gpu_mean"]:
                b[key] = t
        best[g] = b
    return best


def fig_throughput(table, runs, model="9B") -> str | None:
    best = best_per_framework(table, model)
    if not best:
        return None
    layouts = sorted(best)
    order = [k for k in ("pk_warm", "pk_cold", "swift", "unsloth", "hfx", "hf") if any(k in best[g] for g in layouts)]
    xmax = max(t["tokens_per_s_gpu_max"] for g in layouts for t in best[g].values()) * 1.04
    ticks = list(range(0, int(xmax) + 1, 4000))
    BW, VW = 372, 112
    X = lambda v: v / xmax * BW  # noqa: E731
    g1 = best[min(layouts)]
    fastest = max((t for k, t in g1.items() if not k.startswith("pk")), key=lambda t: t["tokens_per_s_gpu_mean"])
    pk1 = g1["pk_warm"]["tokens_per_s_gpu_mean"]
    cols = f"330px {' '.join(f'{BW}px {VW}px' for _ in layouts)}"
    gap = "column-gap:22px"

    def gridlines(h):
        return "".join(f"<div style='position:absolute;left:{X(v):.1f}px;top:0;height:{h}px;width:1px;background:{GRID}'></div>"
                       for v in ticks)

    head = "<div class='mono' style='font-size:12px;color:var(--mute)'>Trainer &middot; best tuned setting</div>"
    for g in layouts:
        head += (f"<div class='mono' style='font-size:12px;color:var(--ink)'>{g} GH200{'s &middot; one node' if g > 1 else ''}"
                 f"</div><div class='mono' style='font-size:12px;color:var(--mute);text-align:right'>tokens/s/GPU</div>")
    rows = [f"<div style='display:grid;grid-template-columns:{cols};{gap};padding-bottom:8px;"
            f"border-bottom:1.5px solid var(--ink)'>{head}</div>"]
    RH = 60
    for k in order:
        fw = k.split("_")[0]
        warm, cold = k == "pk_warm", k == "pk_cold"
        name = NAMES[fw] + (" &middot; warm kernel cache" if warm else " &middot; first process on a node (JIT)" if cold else "")
        cfgs = sorted({best[g][k]["config"] for g in layouts if k in best[g]})
        col = "var(--red)" if fw == "pk" else "var(--ink)"
        cells = (f"<div><div style=\"font:{800 if warm else 600} 15.5px/1.2 'Inter Tight';color:{col}\">{name}</div>"
                 f"<div class='num' style='font-size:12px;color:var(--mute);margin-top:4px'>"
                 f"{' / '.join(cfg_label(c) for c in cfgs)}</div></div>")
        for g in layouts:
            t = best[g].get(k)
            if t is None:
                cells += (f"<div style='position:relative;height:{RH}px'>{gridlines(RH)}"
                          f"<span class='mono' style='position:absolute;left:8px;top:22px;font-size:11.5px;color:var(--mute);"
                          f"background:var(--paper);padding:0 4px'>not run at {g} GPUs</span></div><div></div>")
                continue
            per = [r["tokens_per_s_gpu"] for r in runs if f"{r['run']}/{r['arm']}" in t["runs"]]
            w = X(t["tokens_per_s_gpu_mean"])
            fill = (f"border:1.5px solid {RED};background:repeating-linear-gradient(135deg,transparent 0 4px,{RED} 4px 5.5px)"
                    if cold else f"background:{COLORS[fw]}")
            bar = f"<div style='position:absolute;left:0;top:{RH / 2 - 10:.0f}px;height:20px;width:{w:.1f}px;{fill}'></div>"
            tk = "".join(f"<div style='position:absolute;left:{X(v) - 1:.1f}px;top:{RH / 2 - 16:.0f}px;height:32px;width:2px;"
                         f"background:var(--ink)'></div>" for v in per)
            cells += f"<div style='position:relative;height:{RH}px'>{gridlines(RH)}{bar}{tk}</div>"
            vcol = "color:var(--red);font-weight:500" if warm else ""
            cells += (f"<div style='text-align:right'><div class='num' style='font-size:15px;{vcol}'>"
                      f"{t['tokens_per_s_gpu_mean']:,.0f}</div><div class='num' style='font-size:11.5px;color:var(--mute);"
                      f"margin-top:3px'>n={t['n']}</div></div>")
        rows.append(f"<div style='display:grid;grid-template-columns:{cols};{gap};align-items:center;"
                    f"border-bottom:1px solid var(--faint)'>{cells}</div>")
    axis = "<div></div>"
    for _ in layouts:
        axis += ("<div class='axis' style='position:relative;height:22px'>"
                 + "".join(f"<span style='position:absolute;left:{X(v) - 30:.0f}px;width:60px;text-align:center;top:6px'>"
                           f"{v:,}</span>" for v in ticks) + "</div><div></div>")
    rows.append(f"<div style='display:grid;grid-template-columns:{cols};{gap}'>{axis}</div>")
    rows.append(f"<div style='display:grid;grid-template-columns:{cols};{gap};margin-top:8px'><div></div>"
                f"<div class='axcap' style='grid-column:span {2 * len(layouts)}'>tokens / s / GPU, steps 41-60, real tokens "
                f"(padding excluded) &middot; bar = mean of n jobs, ticks = each job</div></div>")
    legend = ("<div class='legend'>"
              f"<span>{swatch(RED, square=True)} perf_kit, warm kernel cache</span>"
              f"<span>{swatch(RED, hatch=True)} perf_kit, first process on a node (Triton JIT in the window)</span>"
              f"<span>{swatch(INK, square=True)}{swatch(GREYS[1], square=True)}{swatch(GREYS[2], square=True)}"
              f"{swatch(GREYS[3], square=True)} baselines, fastest darkest</span></div>")
    body = (f"<h1>{pk1:,.0f} tokens/s on one GH200: {pk1 / fastest['tokens_per_s_gpu_mean']:.2f}&times; the fastest baseline.</h1>"
            f"<div class='sub' style='max-width:1180px'>Qwen3.5-{model}-Base, LoRA r16 on every projection, identical 64-row steps "
            f"for every framework, each at its best tuned setting from the sweep. The fastest baseline on one GPU is "
            f"{esc(NAMES[fastest['config'].split('_')[0]])}.</div>" + legend + "".join(rows))
    return page(body, "perf_kit &nbsp;/&nbsp; throughput", f"Qwen3.5-{model} &middot; LoRA r16 &middot; GH200", SOURCE)


# ----------------------------------------------------------------------------------------------- loss parity
def fig_loss(runs, job_prefix="main1") -> str | None:
    sel = [r for r in runs if r["run"].startswith(job_prefix) and r["gpus"] == 1]
    by = {r["config"].split("_")[0]: r for r in sel}
    pk = by.get("pk")
    if not pk or not pk.get("step_losses"):
        return None
    ps = {int(k): v for k, v in pk["step_losses"].items()}
    series = {}
    for fw in ("hf", "hfx", "unsloth"):
        r = by.get(fw)
        if r and r.get("step_losses"):
            series[fw] = {int(k): v for k, v in r["step_losses"].items()}
    diffs = {fw: [(x, st[x] - ps[x]) for x in sorted(st) if x in ps] for fw, st in series.items()}
    maxd = {fw: max(abs(d) for _, d in v) for fw, v in diffs.items()}
    tf_max = max(maxd[fw] for fw in ("hf", "hfx") if fw in maxd)
    bound = math.ceil(tf_max / 1e-4)               # the stated bound: the next 1e-4 above the measured maximum
    dash = {"hf": None, "hfx": "7 4", "unsloth": "2 3"}
    line = {"hf": INK, "hfx": GREYS[2], "unsloth": GREYS[1]}   # transformers + PEFT, the parity reference, in black
    allv = list(ps.values()) + [v for st in series.values() for v in st.values()]
    lo, hi = math.floor(min(allv) / 0.05) * 0.05, math.ceil(max(allv) / 0.05) * 0.05
    W = 1030
    top = Plot(W, 330, 0, 61, lo - 0.01, hi + 0.01)
    xs = sorted(ps)
    inner = poly([(x, ps[x]) for x in xs], RED, 4.2, plot=top)
    for fw, st in series.items():
        inner += poly([(x, st[x]) for x in sorted(st)], line[fw], 1.3, dash[fw], plot=top)
    sw = by.get("swift")
    swift_legend = ""
    if sw and sw.get("window_losses"):
        wl = {int(k): v for k, v in sw["window_losses"].items()}
        inner += "".join(f"<rect x='{top.X(x) - 6:.1f}' y='{top.Y(v) - 6:.1f}' width='12' height='12' fill='{INK}' "
                         f"stroke='{PAPER}' stroke-width='2'/>" for x, v in sorted(wl.items()))
        swift_legend = (f"<span style='display:contents'><span>{swatch(INK, square=True)}</span>"
                        f"<span>{esc(NAMES['swift'])}</span><span style='color:var(--mute);grid-column:span 2'>its 10-step "
                        f"means, top only (it trains one extra newline token per row)</span></span>")
    yt = [round(lo + 0.05 * i, 2) for i in range(int(round((hi - lo) / 0.05)) + 1)]
    xt = list(range(0, 61, 10))
    p1 = top.html(inner, xt, yt, lambda v: f"{v}", lambda v: f"{v:.2f}", "", "Loss, mean CE per target",
                  margin="0 0 22px 64px", xlabels=False)
    D = 7e-4
    bot = Plot(W, 190, 0, 61, -D, D)
    inner2 = f"<line x1='0' x2='{W}' y1='{bot.Y(0):.1f}' y2='{bot.Y(0):.1f}' stroke='{RED}' stroke-width='2.5'/>"
    for fw, v in diffs.items():
        inner2 += poly(v, line[fw], 1.8, dash[fw], plot=bot)

    def fd(v):
        return "0" if abs(v) < 1e-12 else signed(v / 1e-4, "+.0f") + "e-4"
    over = (f"<span class='mono' style='position:absolute;left:{W + 10}px;top:{bot.Y(0) - 8:.0f}px;font-size:11.5px;"
            f"color:var(--red);white-space:nowrap'>= perf_kit</span>")
    p2 = bot.html(inner2, xt, [-6e-4, -3e-4, 0, 3e-4, 6e-4], lambda v: f"{v}", fd,
                  "Optimizer step (the same 64 rows per step for every framework)", "Difference to perf_kit", over)
    def lrow(sw_, name, setting, extra=""):
        return (f"<span style='display:contents'><span>{sw_}</span><span>{name}</span>"
                f"<span style='color:var(--mute)'>{setting}</span><span style='text-align:right'>{extra}</span></span>")
    legend = ("<div class='legend' style='display:grid;grid-template-columns:30px auto auto auto;gap:7px 26px;"
              "justify-content:start;align-items:center'>"
              + lrow(swatch(RED, 4), "<span class='red'>perf_kit</span>", cfg_label(pk["config"]), "max |&Delta;|")
              + "".join(lrow(swatch(line[fw], 2, dash[fw]), esc(NAMES[fw]), cfg_label(by[fw]["config"]),
                             f"{maxd[fw] / 1e-4:.1f}e-4") for fw in series)
              + swift_legend + "</div>")
    body = (f"<h1>Same math: within {bound}e-4 of transformers + PEFT at every step.</h1>"
            "<div class='sub' style='max-width:1144px'>Per-step training loss on identical batches, one GH200, all 60 steps. "
            "Top: the baselines are drawn as thin lines over perf_kit's red one and stay inside it. Bottom: each padded "
            "baseline minus perf_kit, on a loss of about 0.95.</div>" + legend + p1 + p2)
    return page(body, "perf_kit &nbsp;/&nbsp; loss parity", "Identical batches &middot; 1 GH200",
                SOURCE + f" &middot; job {esc(pk['run'])}")


# ----------------------------------------------------------------------------------------------- ablations
ABL_NAMES = {"no_fla_conv": "PyTorch per-row conv instead of fla causal conv", "no_liger": "PyTorch RMSNorm / add+norm instead of Liger",
             "no_flash_attn": "eager per-row attention instead of flash-attn varlen",
             "no_swiglu_kernel": "PyTorch SwiGLU instead of the fused kernel",
             "no_fla_gnorm": "PyTorch gated RMSNorm instead of fla's", "ckpt_all": "recompute every layer (standard)",
             "no_fused_grad_acc": "bf16 LoRA grads + add instead of fp32 addmm",
             "fa3": "flash-attn 3 instead of 2", "flashqla": "FlashQLA GDN instead of fla",
             "no_fused_gate": "GDN gate in PyTorch instead of inside fla's kernel"}


def ablation_deltas(abl, ref="mt8192"):
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
    return deltas


def fig_ablation(abl, ref="mt8192") -> str | None:
    deltas = ablation_deltas(abl, ref)
    if not deltas:
        return None
    rows = sorted(deltas, key=lambda c: -statistics.mean(deltas[c]))
    lo = min(min(v) for v in deltas.values())
    hi = max(max(v) for v in deltas.values())
    x0, x1 = min(-8, lo - 2), hi + 3
    BW = 620
    X = lambda v: (v - x0) / (x1 - x0) * BW  # noqa: E731
    ticks = [v for v in range(-10, int(x1) + 1, 10) if x0 <= v <= x1]
    cols = f"440px {BW}px 150px"
    RH = 44

    def gridlines(h):
        g = "".join(f"<div style='position:absolute;left:{X(v):.1f}px;top:0;height:{h}px;width:1px;background:{GRID}'></div>"
                    for v in ticks if v != 0)
        return g + f"<div style='position:absolute;left:{X(0) - 1:.1f}px;top:0;height:{h}px;width:2px;background:{RED}'></div>"

    out = [f"<div style='display:grid;grid-template-columns:{cols};column-gap:20px;padding-bottom:8px;"
           f"border-bottom:1.5px solid var(--ink)' class='mono'><span style='font-size:12px;color:var(--mute)'>"
           f"Removed or replaced</span><span style='font-size:12px;position:relative'><span style='position:absolute;"
           f"left:{X(0) - 6:.0f}px;transform:translateX(-100%);color:var(--mute);white-space:nowrap'>&larr; faster</span>"
           f"<span style='position:absolute;left:{X(0) + 8:.0f}px;color:var(--red);white-space:nowrap'>full perf_kit "
           f"&nbsp;&middot;&nbsp; slower &rarr;</span></span><span style='font-size:12px;color:var(--mute);text-align:right'>"
           f"Step time</span></div>"]
    for c in rows:
        m = statistics.mean(deltas[c])
        left, w = (X(0), X(m) - X(0)) if m >= 0 else (X(m), X(0) - X(m))
        bar = (f"<div style='position:absolute;left:{left:.1f}px;top:{RH / 2 - 9:.0f}px;height:18px;width:{w:.1f}px;"
               f"background:{INK if m > 0 else GREYS[3]}'></div>")
        tk = "".join(f"<div style='position:absolute;left:{X(v) - 1:.1f}px;top:{RH / 2 - 14:.0f}px;height:28px;width:2px;"
                     f"background:var(--ink)'></div>" for v in deltas[c])
        n = len(deltas[c])
        out.append(f"<div style='display:grid;grid-template-columns:{cols};column-gap:20px;align-items:center;"
                   f"border-bottom:1px solid var(--faint)'><div style=\"font:500 15px 'Inter Tight'\">{esc(ABL_NAMES[c])}</div>"
                   f"<div style='position:relative;height:{RH}px'>{gridlines(RH)}{bar}{tk}</div>"
                   f"<div style='text-align:right'><span class='num' style='font-size:15px'>{signed(m, '+.1f')}%</span>"
                   f"<span class='num' style='font-size:11.5px;color:var(--mute);display:inline-block;width:42px'>"
                   f"{'n=' + str(n) if n > 1 else ''}</span></div></div>")
    out.append(f"<div style='display:grid;grid-template-columns:{cols};column-gap:20px'><div></div>"
               f"<div class='axis' style='position:relative;height:22px'>"
               + "".join(f"<span style='position:absolute;left:{X(v) - 30:.0f}px;width:60px;text-align:center;top:6px'>"
                         f"{signed(v, '+d') if v else '0'}%</span>" for v in sorted(set(ticks + [0]))) + "</div><div></div></div>")
    out.append(f"<div style='display:grid;grid-template-columns:{cols};column-gap:20px;margin-top:8px'><div></div>"
               f"<div class='axcap' style='grid-column:span 2'>Step time vs full perf_kit (+ = slower) &middot; bar = mean, "
               f"ticks = each job</div></div>")
    worst = rows[0]
    legend = ("<div class='legend'>"
              f"<span>{swatch(INK, square=True)} removing it slows perf_kit down</span>"
              f"<span>{swatch(GREYS[3], square=True)} the alternative is faster</span>"
              f"<span>{swatch(RED, 2)} <span class='red'>full perf_kit, each job's own reference</span></span></div>")
    head = (f"Recompute every layer and each step takes {statistics.mean(deltas[worst]):.0f}% longer."
            if worst == "ckpt_all" else
            f"The biggest piece: {esc(ABL_NAMES[worst])}, {signed(statistics.mean(deltas[worst]), '+.0f')}%.")
    body = (f"<h1>{head}</h1><div class='sub' style='max-width:1250px'>perf_kit with one piece removed or replaced at a "
            "time: the same 16 steps of 64 rows per configuration, median step time of the last 10, one GH200, 8,192-token "
            "micro-batches, against each job's own full-engine run. Per-step losses are unchanged in every case.</div>"
            + legend + "".join(out))
    return page(body, "perf_kit &nbsp;/&nbsp; ablations", "One piece at a time &middot; 1 GH200", SOURCE)


# ----------------------------------------------------------------------------------------------- micro-batch sweep
def fig_sweep(abl) -> str | None:
    def pts(prefix):
        d = defaultdict(list)
        for a in abl:
            if a.get("median_step_s") and a["config"].startswith(prefix) and not a.get("first_in_run"):
                d[a["micro_tokens"]].append(a)
        return sorted(d.items())
    new, old = pts("mt"), pts("old_planner")
    if not new:
        return None
    tps = lambda v: statistics.mean(x["tokens_per_s_gpu"] for x in v)  # noqa: E731
    rec = lambda v: 100 * v[-1]["ckpt_share"]  # noqa: E731
    W, H = 565, 300
    xs = [k for k, _ in new] + [k for k, _ in old]
    x0, x1 = 2048, max(xs) + 1024
    xt = [4096, 8192, 12288, 16384]
    ymax = math.ceil(max(tps(v) for _, v in new + old) / 2000) * 2000 + 2000
    a = Plot(W, H, x0, x1, 0, ymax)
    b = Plot(W, H, x0, x1, 0, 35)

    def dots(plot, series, f, color, sq):
        if sq:
            return "".join(f"<rect x='{plot.X(k) - 7:.1f}' y='{plot.Y(f(v)) - 7:.1f}' width='14' height='14' fill='{color}' "
                           f"stroke='{PAPER}' stroke-width='2'/>" for k, v in series)
        return "".join(f"<circle cx='{plot.X(k):.1f}' cy='{plot.Y(f(v)):.1f}' r='6.5' fill='{color}' stroke='{PAPER}' "
                       f"stroke-width='2'/>" for k, v in series)

    def label(plot, k, val, text, color, dy, dx=10, anchor="left"):
        pos = f"left:{plot.X(k) + dx:.0f}px" if anchor == "left" else f"right:{plot.W - plot.X(k) + dx:.0f}px"
        return (f"<span class='num' style='position:absolute;{pos};top:{plot.Y(val) + dy:.0f}px;font-size:12px;"
                f"color:{color};white-space:nowrap;background:var(--paper);padding:0 2px'>{text}</span>")

    ia = poly([(k, tps(v)) for k, v in new], RED, 3, plot=a) + dots(a, new, tps, RED, False) + dots(a, old, tps, INK, True)
    ib = poly([(k, rec(v)) for k, v in new], RED, 3, plot=b) + dots(b, new, rec, RED, False) + dots(b, old, rec, INK, True)
    la = lb = ""
    newd, oldd = dict(new), dict(old)
    for k in sorted(set(newd) & set(oldd)):
        la += label(a, k, tps(newd[k]), f"{tps(newd[k]):,.0f}", RED, -32, 0, "right" if k == max(oldd) else "left")
        la += label(a, k, tps(oldd[k]), f"{tps(oldd[k]):,.0f}", INK, 12, 0, "right" if k == max(oldd) else "left")
        lb += label(b, k, rec(oldd[k]), f"{rec(oldd[k]):.1f}%", INK, -8, 14, "left" if k != max(oldd) else "right")
    kmax = max(newd)
    lb += label(b, kmax, rec(newd[kmax]), f"{rec(newd[kmax]):.1f}%", RED, -8, 14, "right")
    fx = lambda v: f"{v:,}"  # noqa: E731
    pa = a.html(ia, xt, list(range(0, ymax + 1, 2000)), fx, lambda v: f"{v:,}", "Micro-batch token budget",
                "Tokens / s / GPU, median step", la, margin="0 0 48px 70px")
    pb = b.html(ib, xt, list(range(0, 36, 5)), fx, lambda v: f"{v}%", "Micro-batch token budget",
                "Layers recomputed", lb, margin="0 0 48px 52px")
    k12 = max(oldd)
    legend = ("<div class='legend'>"
              f"<span>{swatch(RED, 3)} <span class='red'>perf_kit: activation size measured at start-up</span></span>"
              f"<span>{swatch(INK, square=True)} old fixed estimate (0.25 MiB/token/layer)</span></div>")
    body = (f"<h1>Measured activations: {rec(newd[k12]):.0f}% recompute at {k12:,} tokens, not {rec(oldd[k12]):.1f}%.</h1>"
            "<div class='sub' style='max-width:1220px'>perf_kit decides per micro-batch how many layers to recompute so "
            "that stored activations fit the free HBM. Left: throughput against the micro-batch token budget, one GH200. "
            "Right: the share of layers the planner chose to recompute.</div>" + legend
            + f"<div style='display:flex;gap:60px'>{pa}{pb}</div>")
    return page(body, "perf_kit &nbsp;/&nbsp; recompute planner", "Micro-batch sweep &middot; 1 GH200", SOURCE)


# ----------------------------------------------------------------------------------------------- numbers strip
def tps_row(rows, gpus, config, cache="n/a"):
    for r in rows:
        if r["model"] == "9B" and r["gpus"] == gpus and r["config"] == config and r["triton_cache"] == cache:
            return r
    raise KeyError((gpus, config, cache))


def fig_numbers(table, runs) -> str:
    pk = tps_row(table, 1, "pk_12288", "warm")
    pk4 = tps_row(table, 4, "pk_12288", "warm")
    sp = {cfg: pk["tokens_per_s_gpu_mean"] / tps_row(table, 1, cfg)["tokens_per_s_gpu_mean"]
          for cfg in ("swift_8_nockpt", "unsloth_8", "hf_8")}
    sel = {r["config"].split("_")[0]: r for r in runs if r["run"].startswith("main1") and r["gpus"] == 1}
    ps = {int(k): v for k, v in sel["pk"]["step_losses"].items()}
    tf = max(abs(v - ps[int(k)]) for fw in ("hf", "hfx") for k, v in sel[fw]["step_losses"].items() if int(k) in ps)
    bound = math.ceil(tf / 1e-4)
    cells = [("Speed-up &middot; one GH200", f"{sp['swift_8_nockpt']:.2f}&times;", True,
              f"vs ms-swift, the fastest baseline &middot; {sp['unsloth_8']:.2f}&times; vs Unsloth &middot; "
              f"{sp['hf_8']:.2f}&times; vs transformers + PEFT"),
             ("Tokens/s &middot; Qwen3.5-9B LoRA", f"{pk['tokens_per_s_gpu_mean']:,.0f}", False,
              f"per GH200, steady state; {pk4['tokens_per_s_gpu_mean']:,.0f} per GPU on a 4-GPU node"),
             ("Model FLOPs utilisation", f"{100 * pk['mfu_mean']:.1f}%", False,
              f"against 989 TFLOPS; {100 * pk['mfu_clock_adjusted_mean']:.1f}% clock-adjusted, at the clock the "
              f"power-capped GPU ran"),
             ("Loss parity", f"&lt;{bound}e-4", False,
              "per-step loss difference vs transformers + PEFT on identical batches")]
    lab = "".join(f"<div class='mono' style='font-size:12px;color:var(--mute);padding:4px 26px 16px;"
                  f"{'border-left:1.5px solid var(--ink)' if k else ''}'>{c[0]}</div>" for k, c in enumerate(cells))
    big = "".join(f"<div style=\"font:800 64px 'Inter Tight';letter-spacing:-.035em;white-space:nowrap;"
                  f"text-box:trim-both cap alphabetic;padding:0 26px;border-top:1.5px solid var(--ink);"
                  f"border-bottom:1.5px solid var(--ink);{'border-left:1.5px solid var(--ink);' if k else ''}"
                  f"color:{'var(--red)' if c[2] else 'var(--ink)'}\">{c[1]}</div>" for k, c in enumerate(cells))
    sub = "".join(f"<div style='font:400 14px/1.35 IBM Plex Sans;color:#34312B;padding:14px 26px 4px;"
                  f"{'border-left:1.5px solid var(--ink)' if k else ''}'>{c[3]}</div>" for k, c in enumerate(cells))
    body = f"<div style='display:grid;grid-template-columns:repeat(4,316px);margin-top:30px'>{lab}{big}{sub}</div>"
    return page(body, lockup(30), "Perf_Kit &nbsp;&middot;&nbsp; release / 0.1.0",
                "Qwen3.5-9B-Base, LoRA r16, identical 64-row steps, each framework at its best setting &middot; "
                "tested on Arrhenius")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary.json"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--assets", default=str(ROOT / "assets"), help="where headline_numbers.png goes")
    ap.add_argument("--report_img", default=str(ROOT / "report" / "img"), help="copies for report/index.html ('' = none)")
    a = ap.parse_args(argv)
    s = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    out = Path(a.out)
    jobs = [(fig_throughput(s["table"], s["runs"]), out / "throughput.png"),
            (fig_loss(s["runs"]), out / "loss_parity.png"),
            (fig_ablation(s["ablations"]), out / "ablation.png"),
            (fig_sweep(s["ablations"]), out / "micro_sweep.png"),
            (fig_numbers(s["table"], s["runs"]), Path(a.assets) / "headline_numbers.png")]
    jobs = [(doc, p) for doc, p in jobs if doc]
    render(jobs)
    if a.report_img and Path(a.report_img).is_dir():
        for _, p in jobs:
            if p.parent == out:
                shutil.copyfile(p, Path(a.report_img) / p.name)
    print(sorted(p.name for _, p in jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
