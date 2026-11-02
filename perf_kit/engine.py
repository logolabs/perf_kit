"""One optimizer step without per-micro-batch host syncs: packed micro-batches, adaptive checkpointing, one fp32
gradient all-reduce, fused AdamW on the flat LoRA master, compute copies refreshed in two kernels.

The model contract (FastQwen35 implements it): `token_ce(packed, ckpt, chunk) -> per-target CE`, `.lora` (a
LoraStore), `.layers` (len = number of checkpointable layers). The loss of a step is sum(CE) / (labelled targets of
the whole step, all ranks), so ranks may run different numbers of micro-batches and the gradient is that of the
step set, however its rows were laid out.
"""
from __future__ import annotations

import math
import threading
import queue
from dataclasses import dataclass

import torch

from .layout import StepLayout
from .packing import Packed, pack


def lr_cosine(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    """Linear warm-up, then cosine to floor * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


@dataclass
class CkptPolicy:
    """Recompute only as many layers as the activation budget needs, spread evenly over the depth.
    act_mb: stored activations (MiB) per token of a NON-checkpointed layer; in_mb: per token of a checkpointed one
    (its saved input). mode: "auto" | "all" | "none".

    The static defaults are conservative guesses. `measure_act_mb` measures both on the real model at start-up (what
    perf_kit.train does by default, then `calibrate`). The old fixed 0.25 over-estimated the Qwen3.5-9B's measured
    ~0.17 by ~50%, so recompute started while ~1/3 of the HBM was still free."""
    n_layers: int
    budget_gb: float
    act_mb: float = 0.25
    in_mb: float = 0.01
    mode: str = "auto"
    calibrated: bool = False

    def calibrate(self, act_mb: float, in_mb: float, margin: float = 1.05) -> "CkptPolicy":
        """Use measured per-token per-layer sizes, times a small safety margin."""
        if not (act_mb > 0 and in_mb >= 0 and act_mb > in_mb):
            raise ValueError(f"implausible activation sizes act_mb={act_mb} in_mb={in_mb}")
        self.act_mb, self.in_mb, self.calibrated = act_mb * margin, in_mb * margin, True
        return self

    def max_tokens_without_recompute(self) -> int:
        return int(self.budget_gb * 1024.0 / (self.act_mb * self.n_layers))

    def plan(self, tokens: int) -> list[bool]:
        n = self.n_layers
        if self.mode == "all":
            return [True] * n
        if self.mode == "none":
            return [False] * n
        free = self.budget_gb * 1024.0
        per_full, per_ck = self.act_mb * tokens, self.in_mb * tokens
        # (n - k) * per_full + k * per_ck <= free  ->  k >= (n * per_full - free) / (per_full - per_ck)
        k = max(0, min(n, math.ceil((n * per_full - free) / max(1e-9, per_full - per_ck))))
        return [(i + 1) * k // n > i * k // n for i in range(n)]


def _saved_bytes(fn, tokens: int) -> int:
    """Bytes of the activation tensors autograd saves while running fn(): unique storages whose first dim is the
    packed token count (weights and per-row metadata excluded). Device-independent; the CPU path of measure_act_mb."""
    seen: dict = {}

    def pack_hook(t):
        if t.dim() > 0 and t.shape[0] == tokens and not isinstance(t, torch.nn.Parameter):
            st = t.untyped_storage()
            seen[st.data_ptr()] = st.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        out = fn()
    del out
    return sum(seen.values())


def measure_act_mb(model, packed: Packed) -> dict:
    """Measure CkptPolicy's inputs on the real model: run the decoder forward (with grad, no backward) on one packed
    micro-batch, once with no layer checkpointed (-> act_mb) and once with every layer checkpointed (-> in_mb), and
    divide the activation memory that stays alive by tokens x layers. CUDA: the caching allocator's count (ground
    truth, includes what custom kernels keep in ctx); CPU: saved-tensor accounting. in_mb is at least the analytic size
    of a layer's saved inputs (residual stream + pending update, model dtype). Returns {"act_mb", "in_mb", "tokens",
    "method"} in MiB per token per layer."""
    n, T = len(model.layers), int(packed.ids.numel())
    cuda = packed.ids.is_cuda
    res = {}
    if cuda:                    # warm-up: kernels' one-time persistent allocations must not count as activations
        with torch.enable_grad():
            del_me = model.hidden(packed, None)
        del del_me
        torch.cuda.synchronize()
    for key, ck in (("act_mb", None), ("in_mb", [True] * n)):
        with torch.enable_grad():
            if cuda:
                torch.cuda.synchronize()
                m0 = torch.cuda.memory_allocated()
                h = model.hidden(packed, ck)
                torch.cuda.synchronize()
                used = torch.cuda.memory_allocated() - m0 - h.numel() * h.element_size()
                del h
            else:
                used = _saved_bytes(lambda: model.hidden(packed, ck), T)
        res[key] = max(0.0, used / 2**20 / T / n)
    hidden = model.embed.shape[1]
    res["in_mb"] = max(res["in_mb"], 2 * hidden * model.embed.element_size() / 2**20)
    if cuda:
        torch.cuda.empty_cache()
    res.update(tokens=T, method="cuda_allocator" if cuda else "saved_tensors")
    return res


class Prefetcher:
    """Packs + pins the next micro-batches on a CPU thread and copies them to the GPU on a side stream, so the
    training stream never waits on host work."""

    def __init__(self, fetch, device, depth: int = 3):
        self.fetch, self.device = fetch, device
        self.cuda = torch.device(device).type == "cuda"
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.stream = torch.cuda.Stream(device) if self.cuda else None
        self.marked: set = set()                                        # rows whose loss is also tracked apart

    def _make(self, rows) -> Packed:
        p = pack([self.fetch(i) for i in rows], [i in self.marked for i in rows] if self.marked else None)
        if not self.cuda:
            return p
        p = p.pin()
        with torch.cuda.stream(self.stream):
            g = p.to(self.device, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.stream)
        g._ready = ev                                                   # noqa: SLF001
        return g

    def run(self, micro_lists, marked=None):
        """Generator over packed GPU micro-batches for the given row lists (in order); rows in `marked` get
        Packed.mark (their loss is accumulated separately)."""
        items = list(micro_lists)
        self.marked = set(marked) if marked else set()
        out: queue.Queue = queue.Queue(maxsize=3)

        def work():
            try:
                for rows in items:
                    out.put(self._make(rows))
            except BaseException as ex:                                 # noqa: BLE001
                out.put(ex)
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        for _ in items:
            p = out.get()
            if isinstance(p, BaseException):
                raise p
            if self.cuda:
                cur = torch.cuda.current_stream()
                cur.wait_event(p._ready)                                # noqa: SLF001
                for x in (p.ids, p.pos, p.tgt, p.seg, p.cu, p.cu64, p.keep, p.mark):
                    if x is not None:
                        x.record_stream(cur)                            # allocated on the copy stream
            yield p
        worker.join()


class Engine:
    def __init__(self, model, *, lr: float, betas=(0.9, 0.99), eps: float = 1e-8, weight_decay: float = 0.0,
                 clip: float = 1.0, head_chunk: int = 1024, ckpt: CkptPolicy | None = None, world: int = 1,
                 dist=None, segs=(1, 2), fused: bool | None = None, answer_weight: float = 1.0,
                 answer_seg: int | None = None, act_offload=None):
        """answer_weight != 1: targets of segment answer_seg (default: the last of segs) weigh answer_weight in the
        loss, the others 1; the step's normaliser (layout.label_total) must then be the WEIGHTED target count."""
        self.m, self.world, self.dist, self.clip, self.chunk = model, world, dist, clip, head_chunk
        self.answer_weight = float(answer_weight)
        self.answer_seg = segs[-1] if answer_seg is None else answer_seg
        self.ckpt = ckpt
        self.offload = act_offload                                      # perf_kit.offload.ActOffload or None
        model.offload = act_offload
        self.param = torch.nn.Parameter(model.lora.master)              # shares storage with the master buffer
        self.param.grad = model.lora.grad
        dev = model.lora.master.device
        use_fused = (dev.type == "cuda") if fused is None else fused
        self.opt = torch.optim.AdamW([self.param], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                     fused=use_fused)
        self.segs = segs
        self.dev = dev
        # GPU-side running sums (read only at log steps): loss sum, target count, per-segment sum / count
        self.acc = torch.zeros(4 + 2 * len(segs), dtype=torch.float64, device=dev)       # since the last read
        self.acc_step = torch.zeros_like(self.acc)                                         # the current step
        self.ckpt_layers = 0
        self.ckpt_total = 0

    def _accumulate(self, ce: torch.Tensor, packed: Packed):
        seg = packed.seg.index_select(0, packed.keep)
        parts = [ce.sum(), ce.new_full((), float(ce.numel()))]
        for s in self.segs:
            m = (seg == s).to(ce.dtype)
            parts += [(ce * m).sum(), m.sum()]
        if packed.mark is not None:
            mk = packed.mark.index_select(0, packed.keep).to(ce.dtype)
            parts += [(ce * mk).sum(), mk.sum()]
        else:
            parts += [ce.new_zeros(()), ce.new_zeros(())]
        v = torch.stack(parts).to(self.acc.dtype)
        self.acc += v
        self.acc_step += v

    def train_step(self, layout: StepLayout, batches, lr: float):
        """batches: iterable of packed GPU micro-batches (layout.micro, in order). Returns the grad-norm tensor."""
        m = self.m
        for g in self.opt.param_groups:
            g["lr"] = lr
        m.lora.zero_grad()
        self.acc_step.zero_()                                           # after the step: this step's sums only
        inv = 1.0 / max(1e-9, float(layout.label_total)) if self.answer_weight != 1.0 else 1.0 / max(1, layout.label_total)
        for packed in batches:
            plan = self.ckpt.plan(int(packed.ids.numel())) if self.ckpt is not None else None
            if plan is not None:
                self.ckpt_layers += sum(plan)
                self.ckpt_total += len(plan)
            ce = m.token_ce(packed, plan, self.chunk)
            if self.answer_weight != 1.0:
                seg = packed.seg.index_select(0, packed.keep)
                w = torch.where(seg == self.answer_seg, ce.new_full((), self.answer_weight), ce.new_ones(()))
                ((ce * w).sum() * inv).backward()
            else:
                (ce.sum() * inv).backward()
            if self.offload is not None:
                self.offload.end_step()                                 # recycle this micro-batch's host buffers
            self._accumulate(ce.detach(), packed)
        if self.world > 1:
            self.dist.all_reduce(m.lora.grad)                           # SUM: the loss is already globally normalised
        gn = torch.nn.utils.clip_grad_norm_([self.param], self.clip)
        self.opt.step()
        m.lora.push()
        return gn

    @torch.no_grad()
    def eval_ce(self, batches) -> torch.Tensor:
        acc = torch.zeros_like(self.acc)
        keep = self.acc, self.acc_step
        self.acc, self.acc_step = acc, torch.zeros_like(acc)
        try:
            for packed in batches:
                self._accumulate(self.m.token_ce(packed, None, self.chunk), packed)
        finally:
            self.acc, self.acc_step = keep
        if self.world > 1:
            self.dist.all_reduce(acc)
        return acc

    def optim_state(self) -> dict:
        return self.opt.state_dict()

    def load_optim_state(self, sd: dict) -> None:
        self.opt.load_state_dict(sd)

    def read_and_reset(self) -> list[float]:
        """Host copy of the running sums (one sync), then zero them."""
        v = self.acc.tolist()
        self.acc.zero_()
        return v
