"""Turn the raw benchmark logs (results/raw/<run>/...) into results/summary.json and results/summary.md.

Every number in the README / WRITEUP tables comes from here, computed the same way for every framework:

  steady state   the logged windows that end after step `--warm_steps` (default 10): steps 11..N
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
    steady = [w for w in wins if w["step"] - w["win_steps"] >= warm_steps]
    out = {"windows": len(wins), "steady_windows": len(steady)}
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
    ap.add_argument("--warm_steps", type=int, default=10)
    ap.add_argument("--tz", default="Europe/Stockholm", help="time zone of nvidia-smi's timestamps")
    a = ap.parse_args(argv)
    raw, out = Path(a.raw), Path(a.out)
    runs = []
    for job in sorted(p for p in raw.iterdir() if p.is_dir()):
        smi = read_smi(job / "smi.csv", a.tz)
        stamps = read_jsonl(job / "arms.jsonl")
        tasks = next((s["tasks"] for s in stamps if "tasks" in s), 1)
        params = None
        pk = read_jsonl(job / "pk" / "log.jsonl")
        meta = job / "model_params.json"
        if meta.exists():
            params = json.loads(meta.read_text())
        for r in pk:
            if r.get("event") == "model":
                params = {"n_proj": r["frozen_linear_params"], "n_head": r["head_params"], "n_lora": r["lora_params"]}
        for arm_dir in sorted(p for p in job.iterdir() if p.is_dir() and (p / "log.jsonl").exists()):
            if arm_dir.name == "ablate":
                continue
            log = read_jsonl(arm_dir / "log.jsonl")
            if params is None:
                continue
            rec = {"run": job.name, "arm": arm_dir.name, "gpus": tasks, **summarize_arm(log, tasks, params,
                                                                                        a.warm_steps)}
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
        for r in log:
            if r.get("event") == "ablation" and r.get("median_step_s"):
                tps = r["tokens_measured"] / r["seconds_measured"]
                rec = {"run": job.name, "config": r["config"], "micro_tokens": r["micro_tokens"],
                       "median_step_s": r["median_step_s"], "mean_step_s": r["mean_step_s"],
                       "tokens_per_s_gpu": round(tps, 1), "ckpt_share": r["ckpt_share"],
                       "vram_peak_gb": r["vram_peak_gb"], "ckpt_act_mb": r["ckpt_act_mb"],
                       "step_losses": r["step_losses"], "error": r["error"]}
                if mdl and "labels_measured" in r:
                    f = flops(mdl["frozen_linear_params"], mdl["lora_params"], mdl["head_params"],
                              r["tokens_measured"], r["labels_measured"])
                    rec["mfu"] = round(f / (r["seconds_measured"] * PEAK), 4)
                ablations.append(rec)
            elif r.get("event") == "ablation":
                ablations.append({"run": job.name, "config": r["config"], "error": r.get("error") or r.get("skipped")})
    out.mkdir(parents=True, exist_ok=True)
    summary = {"runs": runs, "ablations": ablations,
               "conventions": {"peak_bf16_flops": PEAK, "nominal_sm_mhz": NOMINAL_SM_MHZ,
                               "flops": "(4 N_proj + 6 N_lora) x tokens + 4 N_head x trained targets",
                               "steady_state": f"windows ending after step {a.warm_steps}"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| run | arm | GPUs | tokens/s/GPU | step s | MFU | clock-adj. MFU | SM MHz | peak alloc GB | val (end) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if "tokens_per_s_gpu" not in r:
            lines.append(f"| {r['run']} | {r['arm']} | {r['gpus']} | (no steady windows) | | | | | | |")
            continue
        val = r.get("val") or {}
        vend = val[max(val)] if val else None
        lines.append(f"| {r['run']} | {r['arm']} | {r['gpus']} | {r['tokens_per_s_gpu']:.0f} | {r['step_s']:.3f} | "
                     f"{100 * r['mfu']:.1f}% | {100 * r.get('mfu_clock_adjusted', 0):.1f}% | "
                     f"{r['clock'].get('sm_mhz_median', '')} | {r.get('vram_peak_alloc_gb')} | {vend} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
