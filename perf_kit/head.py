"""Frozen lm_head + cross-entropy, chunked: never materialises the [positions x vocab] logits.

Per chunk of rows: logits = h W^T (bf16 GEMM) -> fp32 -> logsumexp -> CE, and the input gradient
(softmax - onehot) W formed right away for a unit upstream gradient (the head is frozen: no weight gradient), scaled
per token in backward. Peak memory O(chunk x vocab); the same math and dtypes as lm_head + F.cross_entropy(
logits.float()). IGNORE targets give CE 0 and no gradient.
"""
from __future__ import annotations

import torch

IGNORE = -100


class _HeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, tgt, chunk):
        n, acc = h.shape[0], torch.promote_types(w.dtype, torch.float32)
        ce = torch.zeros(n, dtype=acc, device=h.device)
        gh = torch.empty(n, h.shape[1], dtype=w.dtype, device=h.device) if ctx.needs_input_grad[0] else None
        for i in range(0, n, chunk):
            lg = (h[i:i + chunk].to(w.dtype) @ w.t()).to(acc)
            tc = tgt[i:i + chunk]
            valid = tc != IGNORE
            ts = tc.clamp(min=0)
            lse = torch.logsumexp(lg, dim=-1)
            ce[i:i + chunk] = torch.where(valid, lse - lg.gather(1, ts[:, None]).squeeze(1), 0.0)
            if gh is not None:
                pr = lg.sub_(lse[:, None]).exp_()
                pr[torch.arange(pr.shape[0], device=pr.device), ts] -= 1.0
                pr.mul_(valid[:, None])
                gh[i:i + chunk] = pr.to(w.dtype) @ w
            del lg
        ctx.save_for_backward(gh)
        ctx.h_dtype = h.dtype
        return ce

    @staticmethod
    def backward(ctx, gce):
        (gh,) = ctx.saved_tensors
        return (gh.to(gce.dtype) * gce[:, None]).to(ctx.h_dtype), None, None, None


def head_ce(h: torch.Tensor, w: torch.Tensor, tgt: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Per-position CE (fp32) of h [n, d] through the frozen head w [vocab, d] against tgt [n]."""
    return _HeadCE.apply(h, w, tgt, chunk)
