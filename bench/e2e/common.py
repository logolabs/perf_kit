"""Shared pieces of the end-to-end benchmark harnesses (perf_kit, transformers + PEFT, ms-swift).

Every harness trains the SAME token ids (perf_kit.data encodes the rows for all of them; swift's own template is
checked against it), the SAME step sets (step s = rows [s * global_batch, (s + 1) * global_batch) of the file, in
order) and logs the same window records, so tokens/s, MFU and losses are computed one way for all of them
(bench/analysis/aggregate.py):

  {"step": s, "win_steps": k, "win_s": wall seconds of the last k steps (CUDA-synchronised at both ends),
   "win_tokens": real tokens of those steps (all ranks, no padding), "win_labels": trained targets,
   "win_rows": rows, "win_padded_tokens": tokens actually fed incl. padding (padded trainers),
   "step_losses": [per-step mean CE over the step's targets], "t_end": unix time}
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TARGETS = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj")


def dist_env() -> tuple[int, int, int]:
    return tuple(int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))


class JsonLog:
    def __init__(self, out: Path, rank: int):
        self.path, self.rank = Path(out) / "log.jsonl", rank
        if rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, rec: dict):
        if self.rank != 0:
            return
        print(json.dumps(rec), flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def load_rows(model_dir: str, paths, template_kwargs: dict | None, max_len: int, limit: int = 0, cache: str = ""):
    """perf_kit.data rows (ids int32 + offsets + prompt lengths) with the model's tokenizer."""
    from transformers import AutoTokenizer
    from perf_kit import data as D
    tok = AutoTokenizer.from_pretrained(model_dir)
    rows, dropped, src = D.load(list(paths), tok, model_dir, max_len, limit, cache, 0, template_kwargs)
    return tok, rows, dropped, src


def padded_batch(rows, idx: list[int], pad_id: int, device):
    """Right-padded input_ids / attention_mask / labels (HF causal-LM convention: the model shifts the labels) for
    rows `idx` of a perf_kit Rows. Returns (batch dict, real tokens, padded tokens, trained targets)."""
    import torch
    seqs = [rows[i] for i in idx]
    L = max(len(a) for a, _, _ in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    real = n_lab = 0
    for j, (a, b, _) in enumerate(seqs):
        n = len(a)
        ids[j, :n] = torch.from_numpy(a)
        lab[j, :n] = torch.from_numpy(b)
        att[j, :n] = 1
        real += n
        n_lab += int((b[1:] != -100).sum())
    batch = {"input_ids": ids.to(device, non_blocking=True), "attention_mask": att.to(device, non_blocking=True),
             "labels": lab.to(device, non_blocking=True)}
    return batch, real, len(seqs) * L, n_lab


def now() -> float:
    return time.time()
