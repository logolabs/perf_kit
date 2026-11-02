"""Variable-length packing: rows -> one sequence + cu_seqlens, with targets that never cross a row boundary.

A packed micro-batch is what flash-attention's varlen kernels, fla's linear-attention kernels and fla's causal conv
take (batch 1, `cu_seqlens` marking the rows). Position ids restart at 0 in every row, so rotary embeddings match
the unpacked rows. `tgt[t]` is the label the logits at position t must predict: the row's next token (its label,
IGNORE over the prompt), and IGNORE at a row's last position -- so no row ever learns to predict its neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

IGNORE = -100


@dataclass
class Packed:
    ids: torch.Tensor          # [T] int64
    pos: torch.Tensor          # [T] int64, restarting at 0 per row
    tgt: torch.Tensor          # [T] int64, label for the logits at t (IGNORE where none)
    seg: torch.Tensor          # [T] int64, segment id of tgt (callers' convention, e.g. THINK / ANSWER)
    cu: torch.Tensor           # [n+1] int32 cumulative row lengths (flash-attn varlen)
    cu64: torch.Tensor         # [n+1] int64, the same (fla kernels)
    cu_cpu: torch.Tensor       # int64 on the CPU (lets fla skip a device->host copy)
    max_len: int
    keep: torch.Tensor         # [n_keep] int64 positions with a target (the head runs only there)
    n_labels: int
    mark: torch.Tensor | None = None   # [T] float, 1 at positions of rows the caller marked (e.g. a reference rank's)

    def to(self, device, non_blocking: bool = True) -> "Packed":
        mv = lambda t: t.to(device, non_blocking=non_blocking)       # noqa: E731
        return Packed(mv(self.ids), mv(self.pos), mv(self.tgt), mv(self.seg), mv(self.cu), mv(self.cu64),
                      self.cu_cpu, self.max_len, mv(self.keep), self.n_labels,
                      mv(self.mark) if self.mark is not None else None)

    def pin(self) -> "Packed":
        pn = lambda t: t.pin_memory()                                 # noqa: E731
        return Packed(pn(self.ids), pn(self.pos), pn(self.tgt), pn(self.seg), pn(self.cu), pn(self.cu64),
                      self.cu_cpu, self.max_len, pn(self.keep), self.n_labels,
                      pn(self.mark) if self.mark is not None else None)


def pack(rows: list[tuple], marks: list[bool] | None = None) -> Packed:
    """rows: (ids, labels, segs) per row (sequences of ints or numpy arrays; labels IGNORE where not trained, the
    same layout as an unpacked causal-LM batch whose logits at t predict labels[t + 1]). marks: optional flag per
    row -> Packed.mark."""
    lens = [len(r[0]) for r in rows]
    T = int(sum(lens))
    ids = np.empty(T, np.int64)
    pos = np.empty(T, np.int64)
    tgt = np.full(T, IGNORE, np.int64)
    seg = np.zeros(T, np.int64)
    o = 0
    for (a, lab, sg), n in zip(rows, lens):
        ids[o:o + n] = a
        pos[o:o + n] = np.arange(n)
        lab = np.asarray(lab, np.int64)
        tgt[o:o + n - 1] = lab[1:]
        seg[o:o + n - 1] = np.asarray(sg, np.int64)[1:]
        o += n
    cu = np.zeros(len(rows) + 1, np.int64)
    cu[1:] = np.cumsum(lens)
    keep = np.nonzero(tgt != IGNORE)[0]
    mark = None
    if marks is not None:
        mk = np.zeros(T, np.float32)
        for (s0, e0), f in zip(zip(cu[:-1], cu[1:]), marks):
            if f:
                mk[s0:e0] = 1.0
        mark = torch.from_numpy(mk)
    return Packed(torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(tgt), torch.from_numpy(seg),
                  torch.from_numpy(cu.astype(np.int32)), torch.from_numpy(cu.copy()), torch.from_numpy(cu.copy()),
                  max(lens) if lens else 0, torch.from_numpy(keep.astype(np.int64)), int(len(keep)), mark)
