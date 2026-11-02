"""Turn the raw benchmark logs (results/raw/<run>/...) into results/summary.json and results/summary.md.

Every number in the README / WRITEUP tables comes from here, computed the same way for every framework:

  steady state   the logged windows after step `--warm_steps` (default 40): steps 41..60. Triton / inductor JIT
                 compilation and autotuning (perf_kit, ms-swift, Unsloth) can take tens of steps in a cold process,
                 so the table reports perf_kit with a warm kernel cache (an earlier perf_kit process on the node) and
                 cold separately; "first run" columns use steps 11..60 of each run
  tokens/s/GPU   sum(real tokens) / sum(window seconds) / GPUs over the steady windows (padding not counted)
  step time      sum(window seconds) / steps, steady windows
  MFU            model FLOPs / (seconds x GPUs x 989e12), with the LoRA-training convention
                   FLOPs = (4 N_proj + 6 N_lora) x tokens + 4 N_head x trained targets
                 N_proj = frozen projection weights applied to every token (forward 2N + input-gradient 2N; a
                 frozen weight needs no weight gradient), N_lora = trainable LoRA weights (2N + 2N + 2N), N_head =
                 the LM head, needed only where a target exists. Attention-score and Gated-DeltaNet state FLOPs
                 are NOT counted (they are small at these lengths), so MFU is a slight under-estimate for all arms.
  clock-adj MFU  MFU x 1980 MHz / median SM clock in the steady windows (nvidia-smi, 1 Hz): the efficiency at
                 the clock the power-capped GPU actually ran, against the clock the 989 TF peak assumes
  peak VRAM      torch.cuda.max_memory_allocated (per rank, max over ranks) and nvidia-smi memory.used (max)

    python aggregate.py --raw results/raw --out results
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PEAK = 989e12
NOMINAL_SM_MHZ = 1980.0


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_smi(p: Path, tz: str) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8", errors="replace") as fh:
        rd = csv.reader(fh)
        head = [h.strip() for h in next(rd, [])]
        for r in rd:
            if len(r) != len(head):
                continue
            d = dict(zip(head, (x.strip() for x in r)))
            try:
                t = datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=ZoneInfo(tz))
                rec = {"t": t.timestamp(), "gpu": int(d["index"]), "sm": float(d["clocks.sm"]),
                       "power": float(d["power.draw"]), "mem_mib": float(d["memory.used"]),
                       "util": float(d["utilization.gpu"])}
            except (KeyError, ValueError):
                continue
            if "clocks_event_reasons.sw_power_cap" in d:
                rec["sw_cap"] = d["clocks_event_reasons.sw_power_cap"].lower().startswith("active")
            rows.append(rec)
    return rows


def flops(n_proj, n_lora, n_head, tokens, labels) -> float:
    return (4 * n_proj + 6 * n_lora) * tokens + 4 * n_head * labels


def summarize_arm(log: list[dict], world: int, params: dict, warm_steps: int) -> dict:
    wins = [r for r in log if "win_s" in r and "step" in r]
    last = max((w["step"] for w in wins), default=0)
    warm_steps = min(warm_steps, max(10, last - 20))          # shorter runs: their last 20 steps
    steady = [w for w in wins if w["step"] - w["win_steps"] >= warm_steps]
    out = {"windows": len(wins), "steady_windows": len(steady), "steady_from_step": warm_steps + 1}
    cfg = next((r for r in log if r.get("event") == "config"), {})
    model = next((r for r in log if r.get("event") == "model"), {})
    summ = next((r for r in log if r.get("event") == "summary"), {})
    out["device"] = cfg.get("device")
    out["lora_params"] = model.get("lora_params")
    out["lora_A_abs_sum"] = model.get("lora_A_abs_sum")
    if not steady:
        return out
    s = sum(w["win_s"] for w in steady)
    n = sum(w["win_steps"] for w in steady)
    tok = sum(w["win_tokens"] for w in steady)
    lab = sum(w["win_labels"] for w in steady)
    padded = sum(w.get("win_padded_tokens", w["win_tokens"]) for w in steady)
    f = flops(params["n_proj"], params["n_lora"], params["n_head"], tok, lab)
    per_win = [w["win_tokens"] / w["win_s"] / world for w in steady]
    out.update({
        "steps_measured": n, "seconds_measured": round(s, 3), "tokens_measured": tok, "labels_measured": lab,
        "step_s": round(s / n, 4), "tokens_per_s_gpu": round(tok / s / world, 1),
        "tokens_per_s_gpu_windows": [round(x, 1) for x in per_win],
        "tokens_per_s_gpu_window_median": round(statistics.median(per_win), 1),
        "mfu": round(f / (s * world * PEAK), 4), "padding_share": round(1 - tok / padded, 4) if padded else 0.0,
        "t_start": steady[0]["t_end"] - steady[0]["win_s"] if "t_end" in steady[0] else None,
        "t_end": steady[-1].get("t_end"),
        "vram_peak_alloc_gb": summ.get("vram_peak_alloc_gb") or max((w.get("vram_peak_gb") or 0) for w in wins),
        "vram_peak_reserved_gb": summ.get("vram_peak_reserved_gb"),
    })
    # losses: per step where the harness logs them, else per window mean (ms-swift)
    per_step = {}
    for w in wins:
        sl = w.get("step_losses")
        if sl:
            for j, v in enumerate(sl):
                per_step[w["step"] - len(sl) + 1 + j] = v
    sw = next((r for r in log if r.get("event") == "step_losses"), None)
    if sw:
        out["window_losses"] = {int(k): v for k, v in sw["losses"].items()}
    out["step_losses"] = dict(sorted(per_step.items()))
    out["val"] = {r["step"]: r["val_loss"] for r in log if r.get("event") == "val"}
    return out


def clocks(smi: list[dict], t0, t1) -> dict:
    if not smi or t0 is None or t1 is None:
        return {}
    xs = [r for r in smi if t0 <= r["t"] <= t1 and r["util"] >= 30]
    if not xs:
        return {}
    out = {"sm_mhz_median": statistics.median(r["sm"] for r in xs),
           "power_w_median": statistics.median(r["power"] for r in xs),
           "smi_mem_gb_max": round(max(r["mem_mib"] for r in xs) / 1024, 2), "samples": len(xs),
           "util_median": statistics.median(r["util"] for r in xs)}
    caps = [r["sw_cap"] for r in xs if "sw_cap" in r]
    if caps:
        out["sw_power_cap_share"] = round(sum(caps) / len(caps), 3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--out", default="results")
    ap.add_argument("--warm_steps", type=int, default=40)
    ap.add_argument("--first_steps", type=int, default=10, help="'first run' view: windows after this step")
    ap.add_argument("--tz", default="Europe/Stockholm", help="time zone of nvidia-smi's timestamps")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    runs = []
    for job in sorted(p for p in raw.iterdir() if p.is_dir()):
        smi = read_smi(job / "smi.csv", a.tz)
        stamps = read_jsonl(job / "arms.jsonl")
        tasks = next((s["tasks"] for s in stamps if "tasks" in s), 1)
        params = None
        meta = job / "model_params.json"
        if meta.exists():
            params = json.loads(meta.read_text())
        for d in sorted(job.glob("pk*")):              # the model's parameter counts, from any perf_kit arm
            for r in read_jsonl(d / "log.jsonl"):
                if r.get("event") == "model" and "frozen_linear_params" in r:
                    params = {"n_proj": r["frozen_linear_params"], "n_head": r["head_params"],
                              "n_lora": r["lora_params"]}
        kind = job.name.rsplit("_", 1)[0]
        order = [x["arm"] for x in stamps if x.get("event") == "start"]     # arm start order in the job
        for arm_dir in sorted(p for p in job.iterdir() if p.is_dir() and (p / "log.jsonl").exists()):
            if arm_dir.name == "ablate":
                continue
            log = read_jsonl(arm_dir / "log.jsonl")
            if params is None:
                continue
            m = re.match(r"^(.*?)(?:_r(\d+))?$", arm_dir.name)
            config, rep = m.group(1), int(m.group(2) or 1)
            fw = config.split("_")[0]
            rec = {"run": job.name, "kind": kind, "arm": arm_dir.name, "config": config, "framework": fw,
                   "model": "27B" if params["n_proj"] > 2e10 else "9B", "repeat_in_job": rep,
                   # warm: an earlier perf_kit process in this job already compiled the Triton kernels on this
                   # node (node-local cache); cold: the first one, its windows include JIT compilation
                   "triton_cache": "n/a" if not fw.startswith("pk") else "warm" if rep > 1 or (
                       arm_dir.name in order and any(o.startswith("pk") for o in order[:order.index(arm_dir.name)]))
                   else "cold",
                   "gpus": tasks, **summarize_arm(log, tasks, params, a.warm_steps)}
            first = summarize_arm(log, tasks, params, a.first_steps)
            rec["first_run"] = {k: first.get(k) for k in ("tokens_per_s_gpu", "step_s", "mfu", "steps_measured")}
            ck = clocks(smi, rec.get("t_start"), rec.get("t_end"))
            rec["clock"] = ck
            if ck.get("sm_mhz_median") and rec.get("mfu"):
                rec["mfu_clock_adjusted"] = round(rec["mfu"] * NOMINAL_SM_MHZ / ck["sm_mhz_median"], 4)
            runs.append(rec)
        if params is not None:
            (job / "model_params.json").write_text(json.dumps(params))
    ablations = []
    for job in sorted(p for p in raw.iterdir() if p.is_dir()):
        log = read_jsonl(job / "ablate" / "log.jsonl")
        mdl = next((r for r in log if r.get("event") == "model"), None)
        first_cfg = next((r["config"] for r in log if r.get("event") == "ablation" and r.get("median_step_s")), None)
        seen_first = False
        for r in log:
            if r.get("event") == "ablation" and r.get("median_step_s"):
                # same steps for every config: throughput from the MEDIAN step time (one-off compile steps of a
                # config would otherwise dominate a short mean)
                tps = r["tokens_measured"] / r["steps_measured"] / r["median_step_s"]
                rec = {"run": job.name, "config": r["config"], "micro_tokens": r["micro_tokens"],
                       "median_step_s": r["median_step_s"], "mean_step_s": r["mean_step_s"],
                       "tokens_per_s_gpu": round(tps, 1), "ckpt_share": r["ckpt_share"],
                       "vram_peak_gb": r["vram_peak_gb"], "ckpt_act_mb": r["ckpt_act_mb"],
                       "step_losses": r["step_losses"], "error": r["error"],
                       # the first config of a run absorbs the process's JIT warm-up: excluded from comparisons
                       "first_in_run": not seen_first and r["config"] == first_cfg}
                seen_first = True
                if mdl and "labels_measured" in r:
                    f = flops(mdl["frozen_linear_params"], mdl["lora_params"], mdl["head_params"],
                              r["tokens_measured"], r["labels_measured"])
                    rec["mfu"] = round(f / r["steps_measured"] / r["median_step_s"] / PEAK, 4)
                ablations.append(rec)
            elif r.get("event") == "ablation":
                ablations.append({"run": job.name, "config": r["config"], "error": r.get("error") or r.get("skipped")})
    out.mkdir(parents=True, exist_ok=True)
    groups = {}
    for r in runs:
        if "tokens_per_s_gpu" not in r or r["framework"] == "pkprof" or r["kind"] in ("smoke", "tune1"):
            continue
        groups.setdefault((r["model"], r["gpus"], r["config"], r["triton_cache"]), []).append(r)
    table = []
    for (mdl, g, cfg, cache), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], -statistics.mean(
            x["tokens_per_s_gpu"] for x in kv[1]))):
        t = [x["tokens_per_s_gpu"] for x in rs]
        vend = [x["val"][max(x["val"], key=int)] for x in rs if x.get("val")]
        mfu = [x["mfu"] for x in rs]
        cadj = [x["mfu_clock_adjusted"] for x in rs if x.get("mfu_clock_adjusted")]
        mhz = [x["clock"]["sm_mhz_median"] for x in rs if x["clock"].get("sm_mhz_median")]
        ft = [x["first_run"]["tokens_per_s_gpu"] for x in rs if x["first_run"].get("tokens_per_s_gpu")]
        table.append({"model": mdl, "gpus": g, "config": cfg, "triton_cache": cache, "n": len(rs),
                      "runs": [x["run"] + "/" + x["arm"] for x in rs],
                      "tokens_per_s_gpu_mean": round(statistics.mean(t), 1), "tokens_per_s_gpu_min": min(t),
                      "tokens_per_s_gpu_max": max(t), "step_s_mean": round(statistics.mean(x["step_s"] for x in rs), 4),
                      "mfu_mean": round(statistics.mean(mfu), 4),
                      "mfu_clock_adjusted_mean": round(statistics.mean(cadj), 4) if cadj else None,
                      "sm_mhz_mean": round(statistics.mean(mhz)) if mhz else None,
                      "vram_peak_alloc_gb_max": max((x.get("vram_peak_alloc_gb") or 0) for x in rs),
                      "smi_mem_gb_max": max((x["clock"].get("smi_mem_gb_max") or 0) for x in rs),
                      "first_run_tokens_per_s_gpu_mean": round(statistics.mean(ft), 1) if ft else None,
                      "val_end_mean": round(statistics.mean(vend), 5) if vend else None})
    summary = {"table": table, "runs": runs, "ablations": ablations,
               "conventions": {"peak_bf16_flops": PEAK, "nominal_sm_mhz": NOMINAL_SM_MHZ,
                               "flops": "(4 N_proj + 6 N_lora) x tokens + 4 N_head x trained targets",
                               "steady_state": f"windows after step {a.warm_steps}",
                               "first_run": f"windows after step {a.first_steps}"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| model | GPUs | config | Triton cache | n | tokens/s/GPU mean (min-max) | step s | MFU | clock-adj. MFU |"
             " SM MHz | peak alloc GB | val (end) | first run tok/s/GPU |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        lines.append(f"| {t['model']} | {t['gpus']} | {t['config']} | {t['triton_cache']} | {t['n']} | "
                     f"{t['tokens_per_s_gpu_mean']:,.0f} ({t['tokens_per_s_gpu_min']:,.0f}-{t['tokens_per_s_gpu_max']:,.0f})"
                     f" | {t['step_s_mean']:.3f} | {100 * t['mfu_mean']:.1f}% | "
                     f"{100 * (t['mfu_clock_adjusted_mean'] or 0):.1f}% | {t['sm_mhz_mean']} | "
                     f"{t['vram_peak_alloc_gb_max']} | {t['val_end_mean']} | {t['first_run_tokens_per_s_gpu_mean']} |")
    lines += ["", "Per run:", "",
              "| run | arm | GPUs | tokens/s/GPU | step s | MFU | clock-adj. MFU | SM MHz | peak alloc GB | val (end) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if "tokens_per_s_gpu" not in r:
            lines.append(f"| {r['run']} | {r['arm']} | {r['gpus']} | (no steady windows) | | | | | | |")
            continue
        val = r.get("val") or {}
        vend = val[max(val, key=int)] if val else None
        lines.append(f"| {r['run']} | {r['arm']} | {r['gpus']} | {r['tokens_per_s_gpu']:.0f} | {r['step_s']:.3f} | "
                     f"{100 * r['mfu']:.1f}% | {100 * r.get('mfu_clock_adjusted', 0):.1f}% | "
                     f"{r['clock'].get('sm_mhz_median', '')} | {r.get('vram_peak_alloc_gb')} | {vend} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
