"""Activation offload to host memory (Grace LPDDR5X over NVLink-C2C on GH200), with prefetch in backward.

Each decoder layer's forward runs inside `ActOffload.layer(i, tokens)`. Through torch's saved-tensor hooks, every
tensor autograd saves that is an activation (first dim = the micro-batch's packed token count, >= min_bytes) is copied
to pinned host memory on a side stream and dropped from the GPU; weights and small tensors stay where they are. In
backward, the first unpack of layer i starts the host-to-device copies of layer i-1 on another side stream, so the
transfer overlaps layer i's backward; the last `keep_last` layers are never offloaded (their backward starts at once).
The copies are exact, so results are bit-identical with and without offload.

Freed HBM pays for no recompute, bf16 frozen weights (no per-use FP8 dequant) and larger micro-batches. Host memory:
~act_mb x tokens x (layers - keep_last) per GPU (27B at 3,600 tokens: ~40 GB of a GH200's ~120 GB). Pinned buffers
are pooled by size and reused across micro-batches.

    off = ActOffload(n_layers, keep_last=4)
    for i, layer in enumerate(layers):
        with off.layer(i, tokens):
            h = layer(h)
    loss.backward()
    off.end_step()           # after backward: returns buffers to the pool, checks nothing is left in flight
"""
from __future__ import annotations

import contextlib
from collections import defaultdict

import torch


class _Handle:
    __slots__ = ("layer", "host", "event", "gpu", "dtype", "shape", "device")

    def __init__(self, layer, host, event, dtype, shape, device):
        self.layer, self.host, self.event, self.gpu = layer, host, event, None
        self.dtype, self.shape, self.device = dtype, shape, device


class ActOffload:
    def __init__(self, n_layers: int, keep_last: int = 4, min_bytes: int = 1 << 20, enabled: bool = True):
        self.n, self.keep_last, self.min_bytes, self.enabled = n_layers, keep_last, min_bytes, enabled
        self.handles: dict[int, list[_Handle]] = defaultdict(list)
        self.prefetched: set[int] = set()
        self.pool: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        self.d2h = self.h2d = None
        self.bytes_out = 0                 # this step: bytes offloaded (telemetry)
        self.peak_host = 0

    # ---------------------------------------------------------------- buffers / streams
    def _streams(self, dev):
        if self.d2h is None:
            self.d2h, self.h2d = torch.cuda.Stream(dev), torch.cuda.Stream(dev)

    def _buf(self, t: torch.Tensor) -> torch.Tensor:
        key = (t.dtype, t.numel())
        if self.pool[key]:
            return self.pool[key].pop()
        return torch.empty(t.numel(), dtype=t.dtype, pin_memory=True)

    # ---------------------------------------------------------------- hooks
    def _pack(self, layer: int, tokens: int):
        def pack(t: torch.Tensor):
            if (not t.is_cuda or t.dim() == 0 or t.shape[0] != tokens or isinstance(t, torch.nn.Parameter)
                    or t.numel() * t.element_size() < self.min_bytes or not t.is_contiguous()):
                return t
            self._streams(t.device)
            cur = torch.cuda.current_stream(t.device)
            host = self._buf(t)
            self.d2h.wait_stream(cur)                       # the producer kernel must have finished
            with torch.cuda.stream(self.d2h):
                host.copy_(t.reshape(-1), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.d2h)
            t.record_stream(self.d2h)                       # keep t's memory until the copy is done
            h = _Handle(layer, host, ev, t.dtype, t.shape, t.device)
            self.handles[layer].append(h)
            self.bytes_out += host.numel() * host.element_size()
            self.peak_host = max(self.peak_host, self.bytes_out)
            return h
        return pack

    def _fetch(self, layer: int):
        """Start the host-to-device copies of one layer's offloaded tensors on the h2d stream."""
        if layer < 0 or layer in self.prefetched:
            return
        self.prefetched.add(layer)
        for h in self.handles.get(layer, ()):
            self.h2d.wait_event(h.event)                    # its d2h copy must be complete
            with torch.cuda.stream(self.h2d):
                g = torch.empty(h.shape, dtype=h.dtype, device=h.device)
                g.reshape(-1).copy_(h.host, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.h2d)
            h.gpu = (g, ev)

    def _unpack(self, layer: int):
        def unpack(h):
            if not isinstance(h, _Handle):
                return h
            self._fetch(layer)                              # no-op if prefetched by layer + 1
            self._fetch(layer - 1)                          # overlap the next layer's copies with this backward
            g, ev = h.gpu
            cur = torch.cuda.current_stream(g.device)
            cur.wait_event(ev)
            g.record_stream(cur)
            return g
        return unpack

    # ---------------------------------------------------------------- API
    @contextlib.contextmanager
    def layer(self, i: int, tokens: int):
        if not self.enabled or i >= self.n - self.keep_last or not torch.is_grad_enabled():
            yield
            return
        with torch.autograd.graph.saved_tensors_hooks(self._pack(i, tokens), self._unpack(i)):
            yield

    def end_step(self):
        """After a micro-batch's backward: recycle the host buffers (all copies are complete by then)."""
        if self.h2d is not None:
            self.h2d.synchronize()
            self.d2h.synchronize()
        for hs in self.handles.values():
            for h in hs:
                self.pool[(h.dtype, h.host.numel())].append(h.host)
        self.handles.clear()
        self.prefetched.clear()
        out, self.bytes_out = self.bytes_out, 0
        return out
