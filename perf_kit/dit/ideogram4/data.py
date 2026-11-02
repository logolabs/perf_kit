"""Training data for the Ideogram 4 engine: image + caption pairs, a plug-in hook, or synthetic batches.

Three sources, all yielding the same micro-batch dict (one canvas size per micro-batch):

  {"captions": [str] * B, "pixels": (B, 3, H, W) float in [-1, 1]}          images + captions (encoded on the fly)
  {"text_features": [(n_i, D_llm)] * B, "latents": (B, H/16 * W/16, 128), "grid": (H/16, W/16)}
                                                                              precomputed text features + latents

1. ImageCaptionFolder (--data DIR): DIR/captions.jsonl, one {"image": "relative/path.png", "caption": "..."} per
   line. Each image goes to the canvas bucket (--buckets WxH,...) nearest its aspect ratio and is resized to cover
   it and centre-cropped. Captions are passed to the text encoder verbatim.
2. A hook (--dataset module:function): function(args) returns an object with
       batches(epoch: int, rank: int, world: int, batch_size: int) -> iterator of micro-batch dicts
   with either form above; every rank must yield the same number of micro-batches per epoch.
3. Synthetic (--synthetic): random latents and random text features (throughput measurement; the loss is
   meaningless, no encoder is loaded).

The data order is a pure function of (seed, epoch), so a resumed run skips exactly the micro-batches it has
already trained and continues in the order an uninterrupted run would have used.
"""
from __future__ import annotations

import importlib
import json
import random
from pathlib import Path

import torch

PATCH = 16          # pixels per DiT token side (VAE /8 x 2x2 patchify)


def parse_buckets(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        w, h = (int(v) for v in part.lower().split("x"))
        if w % PATCH or h % PATCH:
            raise ValueError(f"bucket {w}x{h}: both sides must be multiples of {PATCH}")
        out.append((w, h))
    return out


def nearest_bucket(w: int, h: int, buckets: list) -> tuple[int, int]:
    import math
    r = math.log(w / h)
    return min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - r))


def to_canvas(im, wh: tuple[int, int]) -> torch.Tensor:
    """PIL image -> (3, H, W) in [-1, 1]: resized to cover the canvas (bicubic), centre-cropped."""
    from PIL import Image
    W, H = wh
    im = im.convert("RGB")
    s = max(W / im.width, H / im.height)
    nw, nh = max(W, round(im.width * s)), max(H, round(im.height * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - W) // 2, (nh - H) // 2
    im = im.crop((left, top, left + W, top + H))
    x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8).view(H, W, 3).permute(2, 0, 1)
    return x.float() / 127.5 - 1.0


def deal(batches: list, rank: int, world: int) -> list:
    """Every rank gets the same number of micro-batches: rank r takes batches r, r + world, ... of the common
    (shuffled) list, truncated to a multiple of world."""
    n = len(batches) // world * world
    return batches[rank:n:world]


class ImageCaptionFolder:
    def __init__(self, root: str, buckets: list, seed: int = 0, limit: int = 0):
        from PIL import Image
        self.root = Path(root)
        rows = [json.loads(ln) for ln in (self.root / "captions.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        if limit:
            rows = rows[:limit]
        if not rows:
            raise SystemExit(f"{self.root / 'captions.jsonl'}: no rows")
        self.rows, self.buckets, self.seed = rows, buckets, seed
        self.by_bucket: dict = {}
        for i, r in enumerate(rows):
            with Image.open(self.root / r["image"]) as im:
                b = nearest_bucket(im.width, im.height, buckets)
            self.by_bucket.setdefault(b, []).append(i)

    def __len__(self):
        return len(self.rows)

    def epoch_batches(self, epoch: int, batch_size: int) -> list:
        """[(bucket, [row indices])], shuffled by (seed, epoch); incomplete batches are dropped."""
        rng = random.Random(self.seed * 1_000_003 + epoch)
        out = []
        for b in sorted(self.by_bucket):
            idx = list(self.by_bucket[b])
            rng.shuffle(idx)
            out += [(b, idx[i:i + batch_size]) for i in range(0, len(idx) - batch_size + 1, batch_size)]
        rng.shuffle(out)
        return out

    def batches(self, epoch: int, rank: int, world: int, batch_size: int):
        from PIL import Image
        for b, idx in deal(self.epoch_batches(epoch, batch_size), rank, world):
            pix = []
            for i in idx:
                with Image.open(self.root / self.rows[i]["image"]) as im:
                    pix.append(to_canvas(im, b))
            yield {"captions": [self.rows[i]["caption"] for i in idx], "pixels": torch.stack(pix)}


class Synthetic:
    """Random latents (B, gh * gw, 128) and text features ([n_i, d_llm]) for throughput runs; n_i varies per row
    (text_len +- 25%) so the packed layout sees realistic ragged lengths."""

    def __init__(self, buckets: list, d_llm: int, text_len: int = 400, seed: int = 0, epoch_batches: int = 1000):
        self.buckets, self.d_llm, self.text_len, self.seed, self.n = buckets, d_llm, text_len, seed, epoch_batches

    def batches(self, epoch: int, rank: int, world: int, batch_size: int):
        for k in range(self.n):
            g = torch.Generator().manual_seed(((self.seed * 1_000_003 + epoch) * 1_000_003 + k) * 64 + rank)
            w, h = self.buckets[k % len(self.buckets)]
            gh, gw = h // PATCH, w // PATCH
            lens = torch.randint(max(1, int(0.75 * self.text_len)), int(1.25 * self.text_len) + 1, (batch_size,),
                                 generator=g).tolist()
            yield {"text_features": [torch.randn(n, self.d_llm, generator=g) for n in lens],
                   "latents": torch.randn(batch_size, gh * gw, 128, generator=g), "grid": (gh, gw)}


def load_hook(spec: str, args):
    mod, _, fn = spec.partition(":")
    if not fn:
        raise SystemExit(f"--dataset {spec!r}: expected module:function")
    return getattr(importlib.import_module(mod), fn)(args)
