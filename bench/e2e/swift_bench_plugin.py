"""ms-swift external plugin (`--external_plugins swift_bench_plugin.py --callbacks perf_kit_bench`): logs the
benchmark's window records (see common.py) from inside swift's trainer, so swift is measured exactly like the
other arms:

  * tokens and trained targets are counted from the batches swift actually feeds (training_step is wrapped; GPU-side
    counters, read only at window ends), not estimated from the dataset;
  * windows are CUDA-synchronised at both ends; losses come from swift's own log (run it with logging_steps equal to
    BENCH_LOG_EVERY: transformers.Trainer syncs the host at every logging step, so logging_steps 1 would slow it);
  * the val loss is computed the same way as in the other harnesses (token-mean CE over the same val rows, encoded by
    perf_kit.data with the same template kwargs), before step 1 and after the last step;
  * the first micro-batch's token ids are saved, so the token-level parity with the other arms can be checked.

Environment: BENCH_OUT (dir), BENCH_LOG_EVERY (10), BENCH_MODEL, BENCH_VAL, BENCH_VAL_ROWS (256),
BENCH_TEMPLATE_KWARGS (JSON), BENCH_VAL_MICRO_BS (8).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from swift.callbacks import TrainerCallback, callbacks_map

_HERE = next(d for d in (Path(__file__).resolve().parent, Path(sys.argv[0]).resolve().parent,
                         Path(os.environ.get("BENCH_E2E_DIR", "."))) if (d / "common.py").exists())
sys.path.insert(0, str(_HERE))
from common import JsonLog, load_rows, padded_batch  # noqa: E402


def _rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class PerfKitBench(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.every = int(os.environ.get("BENCH_LOG_EVERY", "10"))
        self.out = Path(os.environ["BENCH_OUT"])
        self.rank, self.world = _rank_world()
        self.log = JsonLog(self.out, self.rank)
        self.cnt = None                      # GPU [real tokens, trained targets, rows] since the last window
        self.losses: dict = {}
        self.first_ids = None
        self.t_w = None
        self.n_w = 0
        self.last = 0

    # ------------------------------------------------------------------ token counting
    def _wrap(self):
        orig = self.trainer.training_step
        cb = self

        def training_step(model, inputs, *a, **k):
            ids = inputs["input_ids"]
            if cb.cnt is None:
                cb.cnt = torch.zeros(3, dtype=torch.float64, device=ids.device)
            att = inputs.get("attention_mask")
            one = torch.ones((), dtype=torch.float64, device=ids.device)
            if att is not None and att.dim() == 2 and att.shape == ids.shape and "position_ids" not in inputs:
                real, rows = att.sum().double(), one * ids.shape[0]
            else:                                                      # padding_free: one flat sequence
                real = one * ids.numel()
                pos = inputs.get("position_ids")
                rows = (pos.reshape(-1) == 0).sum().double() if pos is not None and pos.dim() <= 2 else one
            lab = inputs.get("labels")
            n_lab = (lab != -100).sum().double() if lab is not None else one * 0
            cb.cnt += torch.stack([real, n_lab, rows])                 # GPU-side: no host sync per micro-batch
            if cb.first_ids is None and cb.rank == 0:
                cb.first_ids = ids.reshape(-1)[:256].tolist()
            return orig(model, inputs, *a, **k)
        self.trainer.training_step = training_step

    # ------------------------------------------------------------------ val (same metric as the other arms)
    @torch.no_grad()
    def _val(self, step):
        path = os.environ.get("BENCH_VAL")
        if not path:
            return
        if not hasattr(self, "_vrows"):
            kw = os.environ.get("BENCH_TEMPLATE_KWARGS")
            tok, self._vrows, _, _ = load_rows(os.environ["BENCH_MODEL"], [path], json.loads(kw) if kw else None,
                                                int(os.environ.get("BENCH_MAX_LEN", "4096")),
                                                int(os.environ.get("BENCH_VAL_ROWS", "256")))
            self._pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        model = self.trainer.model
        dev = next(p for p in model.parameters()).device
        was = model.training
        model.eval()
        mb = int(os.environ.get("BENCH_VAL_MICRO_BS", "8"))
        idx = sorted(range(self.rank, len(self._vrows), self.world), key=lambda i: -self._vrows.length(i))
        tot = torch.zeros(2, dtype=torch.float64, device=dev)
        for j in range(0, len(idx), mb):
            b, _, _, nl = padded_batch(self._vrows, idx[j:j + mb], self._pad, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**b)
            tot[0] += out.loss.double() * nl
            tot[1] += nl
        if self.world > 1:
            dist.all_reduce(tot)
        model.train(was)
        v = tot.tolist()
        self.log({"event": "val", "step": step, "val_loss": round(v[0] / max(1.0, v[1]), 6), "val_targets": int(v[1])})

    # ------------------------------------------------------------------ callbacks
    def on_train_begin(self, args, state, control, **kw):
        self.log({"event": "config", "harness": "ms-swift", "world": self.world,
                  "per_device_train_batch_size": args.per_device_train_batch_size,
                  "gradient_accumulation_steps": args.gradient_accumulation_steps, "max_steps": args.max_steps,
                  "train_dataloader_shuffle": getattr(args, "train_dataloader_shuffle", None),
                  "gradient_checkpointing": args.gradient_checkpointing,
                  "device": torch.cuda.get_device_name(torch.cuda.current_device())})
        model = self.trainer.model
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        a_sum = float(sum(p.detach().double().abs().sum() for n, p in model.named_parameters() if "lora_A" in n))
        self.log({"event": "model", "class": type(model).__name__, "lora_params": n_tr, "lora_A_abs_sum": a_sum,
                  "lora_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})})
        self._wrap()
        self._val(0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t_w = self.t_run = time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            self.losses[state.global_step] = float(logs["loss"])

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        self.n_w += 1
        if s % self.every == 0 or s == 1 or s == args.max_steps:
            torch.cuda.synchronize()
            el = time.time() - self.t_w
            c = self.cnt.clone() if self.cnt is not None else torch.zeros(3, dtype=torch.float64, device="cuda")
            if self.world > 1:
                dist.all_reduce(c)
            tok, lab, _ = c.tolist()
            # padding-free batches carry no per-row marker swift exposes; rows per step are fixed by the geometry
            rows = args.per_device_train_batch_size * args.gradient_accumulation_steps * self.world * self.n_w
            self.cnt.zero_()
            self.log({"step": s, "step_s": round(el / self.n_w, 4), "tokens_per_s_gpu": round(tok / self.world / el),
                      "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "win_steps": self.n_w,
                      "win_s": round(el, 4), "win_tokens": int(tok), "win_labels": int(lab), "win_rows": int(rows),
                      "win_padded_tokens": int(tok), "t_end": time.time()})
            self.n_w = 0
            self.t_w = time.time()

    def on_train_end(self, args, state, control, **kw):
        run_s = time.time() - self.t_run
        self._val(state.global_step)
        steps = sorted(self.losses)
        self.log({"event": "step_losses", "losses": {str(k): self.losses[k] for k in steps},
                  "note": "swift's logged loss: the mean over the logging window ending at that step"})
        self.log({"event": "summary", "steps": state.global_step, "run_s": round(run_s, 2),
                  "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                  "first_input_ids": self.first_ids})


callbacks_map["perf_kit_bench"] = PerfKitBench
