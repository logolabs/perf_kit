"""EMA of the trainable weights as ONE multi-tensor lerp (torch._foreach_lerp_) instead of one kernel per tensor.

The same lerp formula per element as the per-parameter loop, so the result is bit-identical; what changes is the
number of kernel launches (and host time): one fused launch per chunk of tensors instead of ~600 launches per step.
A FROZEN parameter whose shadow is already bit-equal to it is skipped: lerp(s, s, w) = s exactly, so the loop never
changes it either. check_foreach() compares one update against the loop on the real model; the trainer runs it once
at the first optimizer step and logs the maximum difference (0.0 = bit-identical).
"""
from __future__ import annotations

import copy

import torch


class EMA:
    """fp32 shadow copy of a model."""

    def __init__(self, model: torch.nn.Module, decay: float, foreach: bool = True):
        self.decay, self.foreach = decay, foreach
        self.shadow = copy.deepcopy(model).float().eval().requires_grad_(False)
        self._plan, self._flags = None, None

    def _make_plan(self, model):
        fs, fp, rest, skipped = [], [], [], 0
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            if not p.requires_grad and p.dtype == s.dtype and p.device == s.device and torch.equal(s, p.detach()):
                skipped += 1
            elif p.dtype == s.dtype and p.device == s.device:
                fs.append(s)
                fp.append(p.detach())
            else:                                   # a non-fp32 / off-device parameter: the per-tensor path
                rest.append((s, p))
        return fs, fp, rest, skipped

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if not self.foreach:
            for s, p in zip(self.shadow.parameters(), model.parameters()):
                s.lerp_(p.detach().float(), 1 - self.decay)
            return
        flags = tuple(p.requires_grad for p in model.parameters())
        if self._plan is None or flags != self._flags:      # re-planned whenever something is (un)frozen
            self._plan, self._flags = self._make_plan(model), flags
        fs, fp, rest, _ = self._plan
        if fs:
            torch._foreach_lerp_(fs, fp, 1 - self.decay)
        for s, p in rest:
            s.lerp_(p.detach().float(), 1 - self.decay)

    @torch.no_grad()
    def check_foreach(self, model: torch.nn.Module) -> tuple[int, float, int]:
        """One update through update() vs the per-parameter loop from the same shadow; the shadow is restored.
        -> (tensors, max |loop - foreach|, frozen tensors skipped)."""
        saved = [s.detach().clone() for s in self.shadow.parameters()]
        loop = [t.clone() for t in saved]
        for t, p in zip(loop, model.parameters()):
            t.lerp_(p.detach().float(), 1 - self.decay)
        self.update(model)
        diff = max((float((t - s).abs().max()) for t, s in zip(loop, self.shadow.parameters())), default=0.0)
        for s, v in zip(self.shadow.parameters(), saved):
            s.copy_(v)
        return len(saved), diff, self._plan[3] if self._plan else 0
