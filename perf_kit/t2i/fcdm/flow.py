"""Rectified flow in Agate's convention: t = 0 is NOISE, t = 1 is DATA.

    x_t = (1 - t) * noise + t * x1,  target velocity v = x1 - noise
    sampling: z ~ N(0, I) at t = 0, Euler z += dt * v up to t = 1, classifier-free guidance on v
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_t(n: int, device, kind: str = "logit_normal", mu: float = 0.0, std: float = 1.0,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Drawn on the generator's device (CPU by default, so a seeded run is reproducible), then moved."""
    if kind == "uniform":
        t = torch.rand(n, generator=generator)
    elif kind == "logit_normal":
        t = torch.sigmoid(mu + std * torch.randn(n, generator=generator))
    else:
        raise ValueError(f"unknown timestep distribution {kind!r}")
    return t.to(device)


def interpolate(x1: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    tb = t.view(-1, *([1] * (x1.dim() - 1))).to(x1.dtype)
    return (1 - tb) * noise + tb * x1


def flow_loss(model, x1: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor,
              noise: torch.Tensor) -> torch.Tensor:
    """MSE on the velocity (mean over every element)."""
    v = model(interpolate(x1, noise, t), t, ctx, mask)
    return F.mse_loss(v.float(), (x1 - noise).float())


@torch.no_grad()
def euler_sample(model, noise: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor,
                 uncond_ctx: torch.Tensor | None = None, uncond_mask: torch.Tensor | None = None,
                 steps: int = 50, cfg: float = 3.0) -> torch.Tensor:
    """Agate's sampler: t = i / steps for i in 0..steps-1, z += dt * v, CFG v_u + cfg (v_c - v_u)."""
    z, dt = noise, 1.0 / steps
    use_cfg = cfg > 1.0 and uncond_ctx is not None
    if use_cfg:
        B = z.shape[0]
        if uncond_ctx.shape[0] == 1:
            uncond_ctx, uncond_mask = uncond_ctx.expand(B, -1, -1), uncond_mask.expand(B, -1)
        L = max(ctx.shape[1], uncond_ctx.shape[1])
        pad = lambda a, m: (F.pad(a, (0, 0, 0, L - a.shape[1])), F.pad(m, (0, L - m.shape[1])))  # noqa: E731
        ctx, mask = pad(ctx, mask)
        uncond_ctx, uncond_mask = pad(uncond_ctx, uncond_mask)
        ctx_all, mask_all = torch.cat([ctx, uncond_ctx]), torch.cat([mask, uncond_mask])
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt, device=z.device)
        with torch.autocast(z.device.type, dtype=torch.bfloat16, enabled=z.device.type == "cuda"):
            if use_cfg:
                vc, vu = model(torch.cat([z, z]), torch.cat([t, t]), ctx_all, mask_all).float().chunk(2)
                v = vu + cfg * (vc - vu)
            else:
                v = model(z, t, ctx, mask).float()
        z = z + dt * v
    return z
