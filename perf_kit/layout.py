"""Which rows an optimizer step trains on (the STEP SET) is separate from how they are executed (the LAYOUT).

With the loss normalised by the step's global label count, the gradient depends only on the step set: any
redistribution of those rows over ranks and micro-batches gives the same update up to float summation order. So:

  * `reference_steps` reproduces a classic sampler exactly (length-bucketed micro-batches of `micro_bs` rows,
    interleaved over `world` ranks, `accum` micro-batches per rank per step, each rank's cursor wrapping on its own),
    so a fast run trains on the same rows, step for step, as the run it replaces (bit-comparable loss curves);
  * `balance` deals a step's rows to ranks longest-first onto the least-loaded rank (LPT), so no rank waits for a
    straggler (12-23% of every 4-GPU step on our long-tailed SFT data at fixed micro-batch sizes);
  * `split_micro` packs a rank's rows into as few micro-batches as fit a token budget (first-fit decreasing), each
    then run as ONE packed variable-length sequence (no padding).

Everything here is a pure function of lengths and seeds: every rank computes the same plan, nothing is communicated.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


def micro_batches(lengths: list[int], micro_bs: int, seed: int, bucket: int = 64) -> list[list[int]]:
    """Shuffled micro-batches of exactly micro_bs indices, each drawn from a length-sorted window of
    bucket*micro_bs rows (little padding, random order). A ragged tail is dropped. (The classic
    length-grouped sampler of padded trainers.)"""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    out = []
    w = bucket * micro_bs
    for s in range(0, len(idx), w):
        win = sorted(idx[s: s + w], key=lambda i: lengths[i])
        out += [win[j: j + micro_bs] for j in range(0, len(win) - micro_bs + 1, micro_bs)]
    rng.shuffle(out)
    return out


@dataclass
class StepSource:
    """The per-rank micro-batch streams of a reference sampler; step(s) returns the rows of optimizer step s."""
    mbs: list[list[int]]
    world: int
    accum: int
    streams: list[list[list[int]]] = field(init=False)

    def __post_init__(self):
        self.streams = [self.mbs[r::self.world] for r in range(self.world)]
        if any(not s for s in self.streams):
            raise ValueError("fewer micro-batches than ranks")

    @property
    def steps_per_epoch(self) -> int:
        return len(self.mbs) // (self.world * self.accum)

    def step(self, s: int) -> list[int]:
        rows = []
        for st in self.streams:
            for j in range(self.accum):
                rows += st[(s * self.accum + j) % len(st)]
        return rows


def reference_steps(lengths: list[int], micro_bs: int, world: int, accum: int, seed: int, offset: int = 0,
                    bucket: int = 64) -> StepSource:
    """The step sets of a classic padded trainer with these settings (`offset` shifts row indices, e.g. anneal rows
    stored after the main ones)."""
    mbs = micro_batches(lengths, micro_bs, seed, bucket)
    if offset:
        mbs = [[i + offset for i in b] for b in mbs]
    return StepSource(mbs, world, accum)


@dataclass
class SequentialSteps:
    """Step s trains rows [s * global_batch, (s + 1) * global_batch) of the file, wrapping at the end (a ragged tail
    is skipped). The order a framework with an unshuffled sampler sees, so different trainers can be compared on
    identical step sets."""
    n_rows: int
    global_batch: int

    @property
    def steps_per_epoch(self) -> int:
        return self.n_rows // self.global_batch

    def step(self, s: int) -> list[int]:
        if self.steps_per_epoch < 1:
            raise ValueError("fewer rows than one global batch")
        b = (s % self.steps_per_epoch) * self.global_batch
        return list(range(b, b + self.global_batch))


def row_cost(n_tokens: int, quad: float = 0.0) -> float:
    """Execution cost model of one row: linear in tokens (projections, MLP, linear attention) plus an optional
    quadratic term for softmax attention (quad = attention FLOPs per token^2 relative to the linear FLOPs)."""
    return n_tokens + quad * n_tokens * n_tokens


def balance(rows: list[int], lengths, world: int, quad: float = 0.0) -> list[list[int]]:
    """LPT: rows longest-first, each to the currently least-loaded rank (ties -> lowest rank). Deterministic."""
    order = sorted(rows, key=lambda i: (-lengths[i], i))
    load = [0.0] * world
    out: list[list[int]] = [[] for _ in range(world)]
    for i in order:
        r = min(range(world), key=lambda q: (load[q], q))
        out[r].append(i)
        load[r] += row_cost(lengths[i], quad)
    return out


def split_micro(rows: list[int], lengths, max_tokens: int, max_rows: int = 0) -> list[list[int]]:
    """First-fit decreasing into bins of <= max_tokens tokens (and <= max_rows rows if set); a row longer than
    max_tokens gets a bin of its own. Returns bins in a deterministic order (largest first)."""
    order = sorted(rows, key=lambda i: (-lengths[i], i))
    bins: list[list[int]] = []
    fill: list[int] = []
    for i in order:
        n = lengths[i]
        for b in range(len(bins)):
            if fill[b] + n <= max_tokens and (not max_rows or len(bins[b]) < max_rows):
                bins[b].append(i)
                fill[b] += n
                break
        else:
            bins.append([i])
            fill.append(n)
    return bins


def imbalance(per_rank: list[list[int]], lengths, quad: float = 0.0) -> float:
    """max / mean rank cost (1.0 = perfectly balanced)."""
    loads = [sum(row_cost(lengths[i], quad) for i in r) for r in per_rank]
    mean = sum(loads) / len(loads)
    return max(loads) / mean if mean else 1.0


@dataclass
class StepLayout:
    rows: list[int]                      # the step set (all ranks)
    micro: list[list[int]]               # this rank's micro-batches
    label_total: int                     # labelled targets of the whole step (all ranks): the loss normaliser
    tokens_rank: int                     # tokens this rank processes
    tokens_total: int                    # tokens of the whole step
    imbalance: float


def plan_step(rows: list[int], lengths, n_labels, rank: int, world: int, max_tokens: int, max_rows: int = 0,
              quad: float = 0.0, balanced: bool = True, fixed: list[list[int]] | None = None) -> StepLayout:
    """This rank's micro-batches for a step. balanced=False keeps the reference layout `fixed` (per-rank lists)."""
    per_rank = balance(rows, lengths, world, quad) if balanced or fixed is None else fixed
    mine = per_rank[rank]
    micro = split_micro(mine, lengths, max_tokens, max_rows) if mine else []
    return StepLayout(rows=rows, micro=micro, label_total=sum(n_labels[i] for i in rows),
                      tokens_rank=sum(lengths[i] for i in mine), tokens_total=sum(lengths[i] for i in rows),
                      imbalance=imbalance(per_rank, lengths, quad))
