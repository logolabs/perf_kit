"""Training data for the FCDM trainer: latent shards, a plug-in hook, or synthetic latents.

1. LatentShards (--data DIR): any number of shard pairs
       DIR/<name>.safetensors   tensor "latents" (N, 4, H, W): SD-VAE (stabilityai/sd-vae-ft-mse) latent means,
                                unscaled (any float dtype; multiplied by --vae_scale, 0.18215, at load)
       DIR/<name>.jsonl         N lines {"caption": "..."}, row i = latent i
   held in host memory. `python -m perf_kit.t2i.fcdm.encode` writes them from an image folder.
2. A hook (--dataset module:function): function(args) returns an object with
       steps_per_epoch: int
       batch(epoch, step_in_epoch, rank, world, per_rank) -> (latents (per_rank, 4, H, W) float, scaled; [captions])
3. Synthetic (--synthetic_hw N): seeded noise latents (4, N, N) and generated captions, for throughput runs.

The rows of step s are a pure function of (seed, epoch, s), so a resumed run continues in an uninterrupted run's
order on any number of ranks (rank r takes its slice of the step's global batch).
"""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path

import torch

VAE_SCALE = 0.18215


class LatentShards:
    def __init__(self, root: str, global_batch: int, seed: int = 0, vae_scale: float = VAE_SCALE, limit: int = 0):
        from safetensors.torch import load_file
        files = sorted(Path(root).glob("*.safetensors"))
        if not files:
            raise SystemExit(f"{root}: no *.safetensors latent shards")
        lat, caps = [], []
        for f in files:
            x = load_file(str(f))["latents"]
            c = [json.loads(ln)["caption"] for ln in f.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
            if len(c) != x.shape[0]:
                raise SystemExit(f"{f}: {x.shape[0]} latents but {len(c)} captions")
            lat.append(x)
            caps += c
        self.latents = torch.cat(lat)
        self.captions = caps
        if limit:
            self.latents, self.captions = self.latents[:limit], self.captions[:limit]
        self.scale, self.seed, self.G = vae_scale, seed, global_batch
        self.steps_per_epoch = len(self.captions) // global_batch
        if self.steps_per_epoch == 0:
            raise SystemExit(f"{root}: {len(self.captions)} rows, fewer than one global batch of {global_batch}")
        self._perm: dict = {}

    def _order(self, epoch: int) -> list:
        if epoch not in self._perm:
            self._perm = {epoch: torch.randperm(len(self.captions),
                                                generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch))
                          .tolist()}
        return self._perm[epoch]

    def batch(self, epoch: int, step: int, rank: int, world: int, per_rank: int):
        rows = self._order(epoch)[step * self.G: (step + 1) * self.G][rank * per_rank: (rank + 1) * per_rank]
        idx = torch.tensor(rows)
        return self.latents[idx].float() * self.scale, [self.captions[i] for i in rows]


WORDS = ("a red cube", "two cats", "a lighthouse", "a bowl of ramen", "an astronaut", "a watercolor fox",
         "on a wooden table", "under a starry sky", "in the rain", "next to a blue sphere", "macro photo",
         "studio lighting", "pixel art", "a neon sign that reads OPEN", "three yellow apples", "a vintage car")


class Synthetic:
    """Seeded noise latents and generated captions (about `words` phrases each): the loss is meaningless, the loop
    and its throughput are the real ones."""

    def __init__(self, hw: int, seed: int = 0, words: int = 6, steps_per_epoch: int = 10_000):
        self.hw, self.seed, self.words, self.steps_per_epoch = hw, seed, words, steps_per_epoch

    def batch(self, epoch: int, step: int, rank: int, world: int, per_rank: int):
        k = ((self.seed * 1_000_003 + epoch) * 1_000_003 + step) * 4096 + rank
        g = torch.Generator().manual_seed(k)
        rnd = random.Random(k)
        caps = [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(max(1, self.words // 2), self.words * 2)))
                for _ in range(per_rank)]
        return torch.randn(per_rank, 4, self.hw, self.hw, generator=g), caps


def load_hook(spec: str, args):
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit(f"--dataset {spec!r}: expected module:function")
    return getattr(importlib.import_module(mod), fn)(args)
