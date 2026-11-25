"""The outer update of the pool: pseudo-gradients from workers -> new global weights.

Pure tensor math, no files (store.py does the files). Tested in float64 against two references in tests/test_pool.py:
synchronous DiLoCo (Douillard et al. 2023, arXiv 2311.08105, Algorithm 1: mean pseudo-gradient into Nesterov SGD,
outer lr 0.7, momentum 0.9) and the Delayed Nesterov update of asynchronous DiLoCo (Liu et al. 2024, "Asynchronous
Local-SGD Training for Language Modeling", arXiv 2401.09135, Algorithm 3 with c = 0).

A contribution is g = global_at_base - local_after_H_inner_steps, the samples it was trained on, and the progress
counter (`rounds`) of the global version it started from.

Weight. A contribution counts w = samples / round_samples, where round_samples is the samples one *round* of the
pool processes (by default the sum of samples-per-round over the workers alive right now, so "a round" = one
contribution from every live worker, and a 2-GPU worker counts as two 1-GPU ones). This is the sample-weighted
mean FedAvg uses (McMahan et al. 2017, arXiv 1602.05629).

Two modes, both reduce exactly to synchronous DiLoCo when every live worker contributes once from the same base:
  "dn"       (default) Delayed Nesterov: each arrival moves the weights by lr * w * g right away; the momentum is
             updated, and its look-ahead lr * momentum * m applied, once a full round of weight has accumulated.
             Liu et al. introduced it because applying momentum on every stale arrival over-accelerates.
  "buffered" FedBuff-style (Nguyen et al. 2022, arXiv 2106.06639): nothing moves until a full round has
             accumulated, then one Nesterov step on the weighted mean. Simpler, but workers see staler weights.

Staleness = how many rounds of samples were applied between the contribution's base and now. A contribution
staler than `max_staleness` rounds is dropped (its compute is lost and logged); optionally the rest are scaled by
(1 + staleness)^-0.5 ("poly"). Default: drop at > 4 rounds, no scaling. Why: Liu et al. report the poly scaling gave
only "marginal benefits" on top of Delayed Nesterov, so it is off by default; the hard cut bounds the damage of a
worker that hung or ran on a much slower node, which on a shared cluster is the realistic failure (a normal
contribution in an asynchronous pool is about one round stale, never four).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OuterConfig:
    lr: float = 0.7                 # DiLoCo's outer lr
    momentum: float = 0.9           # DiLoCo's outer Nesterov momentum (0 -> plain sample-weighted averaging)
    mode: str = "dn"                # "dn" | "buffered"
    max_staleness: float = 4.0      # rounds; older contributions are dropped
    staleness_weight: str = "none"  # "none" | "poly"

    def __post_init__(self):
        if self.mode not in ("dn", "buffered"):
            raise ValueError(f"mode must be 'dn' or 'buffered', got {self.mode!r}")
        if self.staleness_weight not in ("none", "poly"):
            raise ValueError(f"staleness_weight must be 'none' or 'poly', got {self.staleness_weight!r}")


@dataclass
class OuterState:
    """Everything the global side needs to resume: weights, momentum, the partly filled round, counters."""
    params: dict[str, torch.Tensor]            # trainable parameters: go through the outer optimizer
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)   # float buffers: sample-weighted average only
    m: dict[str, torch.Tensor] | None = None   # outer momentum (None until the first full round)
    buf: dict[str, torch.Tensor] | None = None  # weighted sum of pseudo-gradients of the current, partial round
    acc: float = 0.0                           # weight accumulated in `buf` (a round is complete at 1)
    rounds: float = 0.0                        # rounds of samples applied so far (the staleness clock)
    samples: int = 0                           # samples applied so far (dropped contributions do not count)
    outer_steps: int = 0                       # completed rounds (momentum updates)
    version: int = 0
    high_water: dict[str, int] = field(default_factory=dict)   # worker id -> last contribution seq applied/dropped

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("params", "buffers", "m", "buf", "acc", "rounds", "samples",
                                              "outer_steps", "version", "high_water")}

    @classmethod
    def from_dict(cls, d: dict) -> "OuterState":
        return cls(**d)


@dataclass
class Contribution:
    worker: str
    seq: int                                   # per-worker counter, strictly increasing
    samples: int
    base_rounds: float                         # OuterState.rounds of the version the worker started from
    base_version: int
    delta: dict[str, torch.Tensor]             # global_at_base - local, for params AND float buffers


@dataclass
class Applied:
    worker: str
    seq: int
    status: str                                # "applied" | "dropped" | "duplicate"
    staleness: float
    weight: float
    samples: int = 0


def _zeros_like(d):
    return {k: torch.zeros_like(v) for k, v in d.items()}


def apply(state: OuterState, contribs: list[Contribution], round_samples: float, cfg: OuterConfig) -> list[Applied]:
    """Fold `contribs` (in arrival order) into `state`, in place. Returns what happened to each."""
    if round_samples <= 0:
        raise ValueError("round_samples must be positive")
    log: list[Applied] = []
    for c in contribs:
        if c.seq <= state.high_water.get(c.worker, -1):
            log.append(Applied(c.worker, c.seq, "duplicate", 0.0, 0.0))
            continue
        state.high_water[c.worker] = c.seq
        tau = max(0.0, state.rounds - c.base_rounds)
        if tau > cfg.max_staleness:
            log.append(Applied(c.worker, c.seq, "dropped", tau, 0.0))
            continue
        w = c.samples / round_samples
        if cfg.staleness_weight == "poly":
            w *= (1.0 + tau) ** -0.5
        if state.buf is None:
            state.buf = _zeros_like(state.params)
        with torch.no_grad():
            for k, p in state.params.items():
                g = c.delta[k].to(p.dtype)
                state.buf[k].add_(g, alpha=w)
                if cfg.mode == "dn":
                    p.sub_(g, alpha=cfg.lr * w)
            for k, b in state.buffers.items():    # running statistics: move toward the contributor by its weight
                if k in c.delta:
                    b.sub_(c.delta[k].to(b.dtype), alpha=min(1.0, w))
        state.acc += w
        state.rounds += c.samples / round_samples
        state.samples += int(c.samples)
        log.append(Applied(c.worker, c.seq, "applied", tau, w, int(c.samples)))
        if state.acc >= 1.0 - 1e-9:
            _close_round(state, cfg)
    return log


def export_params(state: OuterState, cfg: OuterConfig) -> dict[str, torch.Tensor]:
    """The weights to evaluate or ship: the stored weights without the outer Nesterov look-ahead,
    x = theta + lr * momentum * m. For the synchronous update this x is exactly the heavy-ball iterate
    x_{t+1} = x_t - lr * m_{t+1} (torch.optim.SGD(momentum, nesterov=False) on the same pseudo-gradients; tested):
    both sequences have the same limit, but the stored Nesterov point is extrapolated by lr * momentum * m, which on
    the cluster runs swung the validation loss by up to 0.4 bits/byte from one version to the next while x moved
    smoothly. Workers keep training from the stored (look-ahead) weights, as in DiLoCo."""
    if state.m is None or cfg.momentum == 0:
        return state.params
    k = cfg.lr * cfg.momentum
    return {n: p + k * state.m[n] for n, p in state.params.items()}


def _close_round(state: OuterState, cfg: OuterConfig) -> None:
    """A full round of weight has arrived: momentum update on its weighted mean + the Nesterov look-ahead."""
    mean = {k: v / state.acc for k, v in state.buf.items()}
    if state.m is None:
        state.m = _zeros_like(state.params)
    with torch.no_grad():
        for k, p in state.params.items():
            state.m[k].mul_(cfg.momentum).add_(mean[k])
            if cfg.mode == "dn":                  # the mean was already applied arrival by arrival
                p.sub_(state.m[k], alpha=cfg.lr * cfg.momentum)
            else:                                 # Nesterov: lr * (g + mu * m), PyTorch SGD(nesterov=True) form
                p.sub_(mean[k] + cfg.momentum * state.m[k], alpha=cfg.lr)
    state.buf = None
    state.acc = 0.0
    state.outer_steps += 1
