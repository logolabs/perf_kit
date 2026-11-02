"""The Agate generator: FCDM-T2 (a ConvNeXt-style FCDM U-Net steered by a recurrent transformer "thinker").

This is the network of LogoLabs' Agate Preview 001 / 002 (huggingface.co/Logolabs/agate-preview-002), whose code
LogoLabs released under the MIT licence (LICENSE-AGATE in this directory keeps that notice). State-dict names are
unchanged, so a release's generator.safetensors loads strictly (load_release). One addition, off by default:
`steer_upsample_once` (see FCDMThinker2._level), the exact-maths change that made the 256 px training step faster.

  renderer   FCDM blocks (Kwon et al., arXiv 2603.09408): 7x7 depthwise conv -> LayerNorm -> per-PIXEL modulation
             (shift, scale, gate) -> 1x1 expand -> GELU -> GRN -> 1x1 project -> gated residual; 3 levels with
             per-level widths and depths; zero-initialised modulation, so every block starts as the identity.
  thinker    a perceiver reads the noisy latent (2x2 patches, RoPE), then prelude blocks, `loops` x shared core
             blocks and coda blocks (self-attention with 2-D RoPE + cross-attention to the text) produce a grid
             "region map"; each level projects it once to a low-rank steering map its blocks decode from.

Conventions: t = 0 is noise, t = 1 is data, the model predicts the velocity x1 - x0.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Agate Preview 002's generator (the "T40r" preset): 190,929,092 parameters with a 512-wide text encoder (Ettin-68M)
AGATE = dict(widths=(288, 832, 1664), depths=(4, 6, 2), steer_rank=(128, 256, 256), ctx_dim=512, dim=640,
             prelude=3, core=3, coda=2, loops=4, loop_probs=(0.0, 0.0, 0.0, 1.0))
TINY = dict(widths=(32, 64, 128), depths=(1, 1, 1), steer_rank=(16, 32, 32), ctx_dim=32, dim=64, p_dim=64,
            perceiver_depth=1, prelude=1, core=1, coda=1, loops=2, loop_probs=(0.5, 0.5), grid=8, n_global=2)
PRESETS = {"agate": AGATE, "tiny": TINY}


# --------------------------------------------------------------------------- FCDM pieces
class LayerNorm2d(nn.Module):
    """LayerNorm over channels of a (B, C, H, W) map; routed through F.layer_norm so autocast runs it in fp32."""

    def __init__(self, dim: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2), channels_first. Identity at init."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x.float(), dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx.to(x.dtype)) + self.beta + x


class TimestepEmbedder(nn.Module):
    """Sinusoidal t (scaled x1000) -> MLP."""

    def __init__(self, dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


def fourier_coords(h: int, w: int, n_freq: int, device) -> torch.Tensor:
    """(h*w, 4*n_freq) sin/cos of normalised pixel-centre coordinates in [-1, 1]."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    f = (2.0 ** torch.arange(n_freq, device=device, dtype=torch.float32)) * math.pi
    feats = [fn(c.reshape(-1, 1) * f) for c in (xx, yy) for fn in (torch.sin, torch.cos)]
    return torch.cat(feats, 1)


class SpatialModBlock(nn.Module):
    """FCDM block whose shift / scale / gate are per-pixel maps decoded by a 1x1 conv from the steering map m, and
    bilinearly upsampled to the block's resolution when m is coarser. Zero-initialised: starts as the identity."""

    def __init__(self, dim: int, d_mod: int, expansion: int = 3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, affine=False)
        self.pw1 = nn.Conv2d(dim, expansion * dim, 1)
        self.act = nn.GELU()
        self.grn = GRN(expansion * dim)
        self.pw2 = nn.Conv2d(expansion * dim, dim, 1)
        self.mod = nn.Conv2d(d_mod, 3 * dim, 1)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        mod = self.mod(m)
        if mod.shape[-2:] != x.shape[-2:]:
            mod = F.interpolate(mod, size=x.shape[-2:], mode="bilinear", align_corners=False)
        shift, scale, gate = mod.chunk(3, dim=1)
        h = self.norm(self.dwconv(x)) * (1 + scale) + shift
        h = self.pw2(self.grn(self.act(self.pw1(h))))
        return x + gate * h


# --------------------------------------------------------------------------- thinker
def rope_freqs(head_dim: int, device) -> torch.Tensor:
    """head_dim/4 angular frequencies per axis, log-spaced from pi/2 to 8 pi."""
    n = head_dim // 4
    return math.pi * 2.0 ** torch.linspace(-1.0, 3.0, n, device=device, dtype=torch.float32)


def grid_xy(h: int, w: int, device) -> torch.Tensor:
    """(h*w, 2) cell-centre coordinates (x, y) in [-1, 1], row-major like pixel_unshuffle."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h * 2 - 1
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w * 2 - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)


def rope_angles(xy: torch.Tensor, head_dim: int, n_unrotated: int = 0) -> torch.Tensor:
    """(N, head_dim/2) rotation angles for positions xy (N, 2); `n_unrotated` zero rows appended (global tokens)."""
    f = rope_freqs(head_dim, xy.device)
    ang = torch.cat([xy[:, :1] * f, xy[:, 1:] * f], 1)
    if n_unrotated:
        ang = torch.cat([ang, ang.new_zeros(n_unrotated, ang.shape[1])], 0)
    return ang


def apply_rope(x: torch.Tensor, ang: torch.Tensor | None) -> torch.Tensor:
    """Rotate channel pairs of x (B, h, N, d) by ang (N, d/2) in fp32 whatever the autocast state (a bf16 rotation
    scrambles positions)."""
    if ang is None:
        return x
    with torch.autocast(x.device.type, enabled=False):
        xf = x.float()
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        c, s = ang.cos(), ang.sin()
        out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)
    return out.to(x.dtype)


class Attn(nn.Module):
    """Multi-head attention with QK RMSNorm and optional 2D RoPE on either side."""

    def __init__(self, dim: int, kv_dim: int, head_dim: int = 64):
        super().__init__()
        self.h, self.d = dim // head_dim, head_dim
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(kv_dim, 2 * dim)
        self.qn, self.kn = nn.RMSNorm(head_dim), nn.RMSNorm(head_dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, kv, pad=None, ang_q=None, ang_k=None):
        B, N, _ = x.shape
        q = self.q(x).view(B, N, self.h, self.d).transpose(1, 2)
        k, v = self.kv(kv).view(B, kv.shape[1], 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.qn(q), ang_q), apply_rope(self.kn(k), ang_k)
        m = None if pad is None else ~pad[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.out(o.transpose(1, 2).reshape(B, N, -1))


class XBlock2(nn.Module):
    """Pre-LN layer: self-attention (optional, RoPE), cross-attention to a context, MLP."""

    def __init__(self, dim: int, ctx_dim: int | None, self_attn: bool = True, mlp: int = 2):
        super().__init__()
        self.sa = Attn(dim, dim) if self_attn else None
        self.n_sa = nn.LayerNorm(dim) if self_attn else None
        self.xa = Attn(dim, ctx_dim) if ctx_dim else None
        self.n_xa = nn.LayerNorm(dim) if ctx_dim else None
        self.n_ctx = nn.LayerNorm(ctx_dim) if ctx_dim else None
        self.n_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp * dim), nn.GELU(), nn.Linear(mlp * dim, dim))

    def forward(self, q, ctx=None, pad=None, ang_q=None, ang_ctx=None):
        if self.sa is not None:
            h = self.n_sa(q)
            q = q + self.sa(h, h, None, ang_q, ang_q)
        if self.xa is not None:
            c = self.n_ctx(ctx)
            q = q + self.xa(self.n_xa(q), c, pad, ang_q if ang_ctx is not None else None, ang_ctx)
        return q + self.mlp(self.n_mlp(q))


class Thinker2(nn.Module):
    """read (perceiver over 2x2 latent patches, RoPE) -> think (prelude, `loops` x core, coda) -> region map."""

    def __init__(self, ctx_dim: int, latent_ch: int, dim: int, p_dim: int, perceiver_depth: int,
                 prelude: int, core: int, coda: int, loops: int, grid: int, n_global: int, n_freq: int = 6,
                 rope: bool = True):
        super().__init__()
        self.grid, self.n_global, self.n_freq, self.rope = grid, n_global, n_freq, rope
        self.loops = loops
        pos = 4 * n_freq
        self.queries = nn.Parameter(torch.randn(grid * grid + n_global, p_dim) * 0.02)
        self.q_pos = nn.Linear(pos, p_dim)
        self.t1 = TimestepEmbedder(p_dim)
        self.t2 = TimestepEmbedder(dim)
        self.in_latent = nn.Linear(latent_ch * 4 + pos, p_dim)
        self.perceiver = nn.ModuleList(XBlock2(p_dim, p_dim, self_attn=False) for _ in range(perceiver_depth))
        self.lift = nn.Linear(p_dim, dim)
        self.prelude = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(prelude))
        self.core = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(core))
        self.coda = nn.ModuleList(XBlock2(dim, ctx_dim) for _ in range(coda))
        self.adapter = nn.Linear(2 * dim, dim)
        with torch.no_grad():
            self.adapter.weight.zero_()
            self.adapter.weight[:, :dim].copy_(torch.eye(dim))
            self.adapter.bias.zero_()
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, t, ctx, mask, loops: int | None = None):
        B, _, H, W = z.shape
        g, dev = self.grid, z.device
        loops = self.loops if loops is None else loops
        pos = lambda h, w: fourier_coords(h, w, self.n_freq, dev)[None].expand(B, -1, -1)  # noqa: E731
        lat = F.pixel_unshuffle(z, 2).flatten(2).transpose(1, 2)
        kv = self.in_latent(torch.cat([lat, pos(H // 2, W // 2)], -1))
        q = self.queries[None].expand(B, -1, -1).clone()
        q[:, :g * g] = q[:, :g * g] + self.q_pos(pos(g, g))
        q = q + self.t1(t)[:, None]
        ang_pq = ang_pk = ang_q = None
        if self.rope:
            ang_pq = rope_angles(grid_xy(g, g, dev), 64, self.n_global)
            ang_pk = rope_angles(grid_xy(H // 2, W // 2, dev), 64)
            ang_q = ang_pq
        for blk in self.perceiver:
            q = blk(q, kv, None, ang_pq, ang_pk)
        q = self.lift(q) + self.t2(t)[:, None]
        pad = ~mask.bool()
        for blk in self.prelude:
            q = blk(q, ctx, pad, ang_q)
        e, s = q, q
        for _ in range(loops):
            s = self.adapter(torch.cat([s, e], -1))
            for blk in self.core:
                s = blk(s, ctx, pad, ang_q)
        for blk in self.coda:
            s = blk(s, ctx, pad, ang_q)
        out = self.norm(s)
        return out[:, :g * g].transpose(1, 2).reshape(B, -1, g, g)


class FCDMThinker2(nn.Module):
    def __init__(self, latent_ch: int = 4, C: int = 224, depths: tuple = (4, 6, 2), expansion: int = 3,
                 ctx_dim: int = 512, dim: int = 512, p_dim: int = 256, perceiver_depth: int = 4,
                 prelude: int = 2, core: int = 2, coda: int = 2, loops: int = 4,
                 loop_probs: tuple = (0.1, 0.1, 0.2, 0.6), grid: int = 16, n_global: int = 16,
                 rope: bool = True, widths: tuple | None = None, steer_rank: tuple | None = None,
                 grad_checkpoint: bool = False):
        super().__init__()
        depths = tuple(depths)
        widths = tuple(widths) if widths else (C, 2 * C, 4 * C)
        steer_rank = tuple(steer_rank) if steer_rank else (0, 0, 0)
        self.config = dict(latent_ch=latent_ch, C=C, depths=depths, expansion=expansion, ctx_dim=ctx_dim,
                           dim=dim, p_dim=p_dim, perceiver_depth=perceiver_depth, prelude=prelude, core=core,
                           coda=coda, loops=loops, loop_probs=tuple(loop_probs), grid=grid, n_global=n_global,
                           rope=rope, widths=widths, steer_rank=steer_rank)
        if len(loop_probs) != loops or abs(sum(loop_probs) - 1) > 1e-6:
            raise ValueError(f"loop_probs {loop_probs} must give one probability per loop count 1..{loops}")
        self.loop_probs = tuple(loop_probs)
        self.grad_checkpoint = grad_checkpoint
        self.thinker = Thinker2(ctx_dim, latent_ch, dim, p_dim, perceiver_depth, prelude, core, coda, loops,
                                grid, n_global, rope=rope)
        self.loops = loops
        self.steer_upsample_once = False      # set by the trainer (--steer_upsample_once); not a config key
        d0, d1, d2 = depths
        w0, w1, w2 = widths
        self.steer_in = nn.ModuleDict({lvl: nn.Conv2d(dim, r, 1) for lvl, r in zip(("l0", "l1", "l2"), steer_rank)
                                       if r})
        d_in = [r or dim for r in steer_rank]
        self.stem = nn.Conv2d(latent_ch, w0, 3, padding=1)
        mk = lambda w, dm, n: nn.ModuleList(SpatialModBlock(w, dm, expansion) for _ in range(n))  # noqa: E731
        self.enc0, self.enc1, self.mid = mk(w0, d_in[0], d0), mk(w1, d_in[1], d1), mk(w2, d_in[2], 2 * d2)
        self.dec1, self.dec0 = mk(w1, d_in[1], d1), mk(w0, d_in[0], d0)
        self.down0 = nn.Sequential(LayerNorm2d(w0), nn.Conv2d(w0, w1, 2, stride=2))
        self.down1 = nn.Sequential(LayerNorm2d(w1), nn.Conv2d(w1, w2, 2, stride=2))
        self.up1 = nn.ConvTranspose2d(w2, w1, 2, stride=2)
        self.up0 = nn.ConvTranspose2d(w1, w0, 2, stride=2)
        self.final_norm = LayerNorm2d(w0, affine=False)
        self.final_mod = nn.Conv2d(dim, 2 * w0, 1)
        self.final_conv = nn.Conv2d(w0, latent_ch, 3, padding=1)
        for m in (self.final_mod, self.final_conv):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def draw_loops(self, step: int, seed: int = 0) -> int:
        """The loop count for one training step, seeded by the step: every rank (and a resumed run) agrees."""
        r = random.Random(seed * 1_000_003 + step).random()
        acc = 0.0
        for n, p in enumerate(self.loop_probs, 1):
            acc += p
            if r < acc:
                return n
        return len(self.loop_probs)

    @staticmethod
    def _at(m, x):
        return F.adaptive_avg_pool2d(m, x.shape[-2:]) if m.shape[-1] > x.shape[-1] else m

    def _level(self, blocks, x, m, lvl):
        if lvl in self.steer_in:                      # low-rank steering: project once per level
            m = self.steer_in[lvl](m)
        m = self._at(m, x)
        if self.steer_upsample_once and m.shape[-1] < x.shape[-1]:
            # Same function, less memory traffic: each block's 1x1 mod conv commutes with bilinear upsampling (the
            # interpolation weights sum to 1, so the bias passes through), so upsampling the r-channel steering map
            # ONCE per level replaces every block's upsampling of its 3C modulation maps. That upsampling's backward
            # (index_put) was 16% of the 256 px training step's GPU time. Off by default; the result differs from
            # the default path by float rounding only (tested in float64).
            m = F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)
        for blk in blocks:
            if self.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x, m, use_reentrant=False)
            else:
                x = blk(x, m)
        return x

    def forward(self, z, t, ctx, mask, loops: int | None = None):
        """z: (B, 4, H, W), H and W divisible by 4 and by the grid; t: (B,) in [0, 1]; ctx: (B, L, ctx_dim) text
        states; mask: (B, L), 1 = real token; loops: thinking loops (default self.loops). -> velocity."""
        m = self.thinker(z, t, ctx.float(), mask, self.loops if loops is None else loops)
        x0 = self._level(self.enc0, self.stem(z), m, "l0")
        x1 = self._level(self.enc1, self.down0(x0), m, "l1")
        x = self._level(self.mid, self.down1(x1), m, "l2")
        x = self._level(self.dec1, self.up1(x) + x1, m, "l1")
        x = self._level(self.dec0, self.up0(x) + x0, m, "l0")
        fm = self.final_mod(m)
        if fm.shape[-2:] != x.shape[-2:]:
            fm = F.interpolate(fm, size=x.shape[-2:], mode="bilinear", align_corners=False)
        shift, scale = fm.chunk(2, dim=1)
        return self.final_conv(self.final_norm(x) * (1 + scale) + shift)


def build(preset: str = "agate", **overrides) -> FCDMThinker2:
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    return FCDMThinker2(**kw)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def resolve(repo_or_dir: str) -> Path:
    p = Path(repo_or_dir)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_or_dir, allow_patterns=["config.json", "generator.safetensors",
                                                               "text_encoder/*"]))


def load_release(repo_or_dir: str) -> tuple[FCDMThinker2, dict, Path]:
    """An Agate release (config.json + generator.safetensors, e.g. Logolabs/agate-preview-002) -> (generator in
    fp32, its config, the release directory). The text encoder the release was trained with is <dir>/text_encoder."""
    from safetensors.torch import load_file
    root = resolve(repo_or_dir)
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if cfg.get("arch") != "fcdm_t2":
        raise SystemExit(f"{root}: arch {cfg.get('arch')!r}; this trainer supports the fcdm_t2 releases (001, 002)")
    m = FCDMThinker2(**cfg["model_kw"])
    m.load_state_dict(load_file(str(root / "generator.safetensors")))
    return m, cfg, root
