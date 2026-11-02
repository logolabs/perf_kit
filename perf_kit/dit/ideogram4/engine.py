"""Fast LoRA training engine for the Ideogram 4 DiT: perf_kit's techniques on a diffusion transformer.

What it computes is the LoRA training step of the upstream Ideogram4Transformer (perf_kit/dit/ideogram4/upstream,
from Ideogram's Apache-2.0 inference code) with LoRA on attention.qkv / attention.o / feed_forward.w1-3: flow MSE
over image tokens, float32 rotary tables (see stack.restore_rope_precision), rsLoRA or plain alpha / r scale, and
optionally K timestep-binned adapters blended per sample (n_bins > 1; n_bins = 1 is one ordinary LoRA). How:

  * PACKED tokens: the left padding of the upstream layout is dropped; every GEMM and every elementwise op runs over
    real tokens only; attention is flash-attention 3 varlen (cu_seqlens), with a per-sample SDPA reference on CPU.
  * FUSED projections + fused LoRA, no autograd: y = x W^T (one GEMM; w1 | w3 share one GEMM), xa = x A^T for every
    member and every bin at once, the per-sample bin weights applied to xa, y_m += xa_w (s B_m)^T. The backward is
    written out by hand; LoRA gradients are accumulated straight into float32 buffers with
    torch.addmm(acc, a, b, out_dtype=float32, out=acc) (no AccumulateGrad, no casts, no PEFT wrappers).
  * The FP8 checkpoint stays as e4m3 codes + per-row scales (half the memory) and is dequantised bit-exactly into a
    shared scratch buffer right before each GEMM (kernels.dequant_rows_into).
  * Lean activations: per block it keeps the residual input (float32), qkv, the attention output + lse, the o
    output, the fused w1|w3 output and the w2 output (bf16); norms, modulation, rotary and SwiGLU are recomputed in
    the backward (elementwise only). Whole-block recompute is used only for blocks past a memory budget, and the
    trainer can split a micro-batch's DiT pass into sub-batches so that no block is recomputed (train.plan_split).
  * Frozen, gradient-free parts (adaLN modulations of all blocks, text / latent input projections, rotary tables,
    bin weights) are computed once per micro-batch under no_grad; the final layer runs on image tokens only.
  * One flat float32 master / grad / bf16 compute store for every adapter (LoraStore), so the optimizer is one fused
    AdamW, the gradient all-reduce is one call and refreshing the compute copies is two kernels.

Adapters are written in PEFT format when n_bins == 1 (LoraStore.state_peft: loadable with peft onto the upstream
transformer) and in a binned format otherwise (LoraStore.state_binned: base_model.layers.N.<target>.lora_A
[K, r, in] / .lora_B [K, out, r]).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ops
from .kernels import dequant_rows_into, fp8_linear

TARGETS = ("attention.qkv", "attention.o", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
GROUPS = (("qkv", ("attention.qkv",)), ("o", ("attention.o",)), ("w13", ("feed_forward.w1", "feed_forward.w3")),
          ("w2", ("feed_forward.w2",)))
# Ideogram 4's sequence-format constants (ideogram4/constants.py)
OUTPUT_IMAGE_INDICATOR, LLM_TOKEN_INDICATOR, IMAGE_POSITION_OFFSET = 2, 3, 65536


# =========================================================================== frozen weights
class Scratch:
    """One buffer per weight shape, shared by every group of that shape (one stream: dequant -> GEMM in order)."""

    def __init__(self):
        self.bufs: dict = {}

    def get(self, shape, device, dtype):
        k = (tuple(shape), str(device), dtype)
        b = self.bufs.get(k)
        if b is None:
            b = self.bufs[k] = torch.empty(shape, dtype=dtype, device=device)
        return b


@dataclass
class Frozen:
    out: int
    inp: int
    w: torch.Tensor | None = None            # stored in the compute dtype, or
    codes: torch.Tensor | None = None        # e4m3 [out, in] + per-row float32 scale [out]
    scale: torch.Tensor | None = None
    scale_row: torch.Tensor | None = None    # [1, out] view for the FP8 forward GEMM

    def weight(self, scratch: Scratch, dtype) -> torch.Tensor:
        if self.w is not None:
            return self.w
        buf = scratch.get((self.out, self.inp), self.codes.device, dtype)
        return dequant_rows_into(buf, self.codes, self.scale)

    def nbytes(self) -> int:
        if self.w is not None:
            return self.w.numel() * self.w.element_size()
        return self.codes.numel() + self.scale.numel() * 4


def frozen_from(mods: list, cdt, storage: str) -> Frozen:
    """Concatenate the rows of Linears that read the same input (w1 | w3) into one frozen weight. A module with a
    `weight_scale` (the ideogram4 package's Fp8Linear) is FP8: kept as codes with storage="fp8", else dequantised."""
    fp8 = [hasattr(m, "weight_scale") for m in mods]
    out = sum(m.weight.shape[0] for m in mods)
    inp = mods[0].weight.shape[1]
    for m in mods:
        if getattr(m, "bias", None) is not None:
            raise ValueError("biased target Linear: not supported (Ideogram 4 targets have no bias)")
    if all(fp8) and storage == "fp8":
        codes = torch.cat([m.weight for m in mods]).contiguous()
        scale = torch.cat([m.weight_scale.float() for m in mods]).contiguous()
        return Frozen(out, inp, codes=codes, scale=scale, scale_row=scale.view(1, -1))
    parts = []
    for m, is8 in zip(mods, fp8):
        if is8:                                  # the same expression a dequantise-at-load trainer uses
            parts.append((m.weight.to(torch.float32) * m.weight_scale.to(torch.float32).unsqueeze(1)).to(cdt))
        else:
            parts.append(m.weight.detach().to(cdt))
    return Frozen(out, inp, w=torch.cat(parts).contiguous())


# =========================================================================== LoRA (binned) store
@dataclass
class Member:
    name: str            # module path inside the transformer, e.g. layers.3.feed_forward.w1
    out: int
    off: int             # column offset in the group output
    b_master: torch.Tensor | None = None     # engine layout [out, K * r]
    b_grad: torch.Tensor | None = None
    sb_c: torch.Tensor | None = None         # compute copy of s * B


@dataclass
class Group:
    block: int
    kind: str
    frozen: Frozen
    members: list
    inp: int
    a_master: torch.Tensor | None = None     # [M * K * r, in]: member-major, then bin, then rank (= lora_A [K, r, in])
    a_grad: torch.Tensor | None = None
    a_c: torch.Tensor | None = None
    fp8_fwd: bool = False

    @property
    def out(self) -> int:
        return self.frozen.out


class LoraStore:
    """Every adapter in two flat buffers: [all A | all B] masters, the same layout for grads and compute copies."""

    def __init__(self, groups: list, n_ad: int, rank: int, scale: float, device, master_dtype, compute_dtype):
        self.groups, self.K, self.r, self.s = groups, n_ad, rank, scale
        kr = n_ad * rank
        n_a = sum(len(g.members) * kr * g.inp for g in groups)
        n_b = sum(m.out * kr for g in groups for m in g.members)
        self.n_a, self.n_b = n_a, n_b
        self.master = torch.zeros(n_a + n_b, dtype=master_dtype, device=device)
        self.grad = torch.zeros_like(self.master)
        self.compute = torch.zeros(n_a + n_b, dtype=compute_dtype, device=device)
        oa, ob = 0, n_a
        self.members: dict = {}
        for g in groups:
            n = len(g.members) * kr * g.inp
            g.a_master = self.master[oa: oa + n].view(-1, g.inp)
            g.a_grad = self.grad[oa: oa + n].view(-1, g.inp)
            g.a_c = self.compute[oa: oa + n].view(-1, g.inp)
            oa += n
            for m in g.members:
                nb = m.out * kr
                m.b_master = self.master[ob: ob + nb].view(m.out, kr)
                m.b_grad = self.grad[ob: ob + nb].view(m.out, kr)
                m.sb_c = self.compute[ob: ob + nb].view(m.out, kr)
                ob += nb
        for g in groups:
            for j, m in enumerate(g.members):
                self.members[m.name] = (g, j, m)

    @property
    def numel(self) -> int:
        return self.master.numel()

    def push(self):
        """Refresh the compute copies from the masters (after every optimizer step): A as is, B times s."""
        self.compute[: self.n_a].copy_(self.master[: self.n_a])
        torch.mul(self.master[self.n_a:], self.s, out=self.compute[self.n_a:])

    def zero_grad(self):
        self.grad.zero_()

    def init_new(self, seed: int) -> None:
        """PEFT's default LoRA init per bin: A kaiming-uniform(a=sqrt(5)) = U(-1/sqrt(in), 1/sqrt(in)), B = 0."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for g in self.groups:
                bound = 1.0 / math.sqrt(g.inp)
                g.a_master.copy_(torch.empty(g.a_master.shape).uniform_(-bound, bound, generator=gen))
            self.master[self.n_a:].zero_()
        self.push()

    # ---------------------------------------------------- per-adapter views (binned layout)
    def ref_order(self) -> list:
        """(key, kind, member name) per adapter tensor: per block qkv, o, w1, w2, w3, each lora_A then lora_B."""
        out = []
        nb = max(g.block for g in self.groups) + 1
        for i in range(nb):
            for t in TARGETS:
                n = f"layers.{i}.{t}"
                if n in self.members:
                    out.append((f"base_model.{n}.lora_A", "A", n))
                    out.append((f"base_model.{n}.lora_B", "B", n))
        return out

    def to_ref(self, flat: torch.Tensor, name: str, kind: str) -> torch.Tensor:
        """A tensor laid out like `master` (e.g. an Adam moment) -> the view of one adapter tensor:
        A [K, r, in], B [K, out, r]."""
        g, j, m = self.members[name]
        kr = self.K * self.r
        if kind == "A":
            off = (g.a_master.data_ptr() - self.master.data_ptr()) // self.master.element_size()
            n = kr * g.inp
            return flat[off + j * n: off + (j + 1) * n].view(self.K, self.r, g.inp)
        off = (m.b_master.data_ptr() - self.master.data_ptr()) // self.master.element_size()
        return flat[off: off + m.out * kr].view(m.out, self.K, self.r).permute(1, 0, 2)

    def state_binned(self, flat: torch.Tensor | None = None) -> dict:
        flat = self.master if flat is None else flat
        return {k: self.to_ref(flat, n, kind).detach().to("cpu", torch.float32).contiguous()
                for k, kind, n in self.ref_order()}

    def load_into(self, flat: torch.Tensor, sd: dict, strict: bool = True) -> int:
        n = 0
        with torch.no_grad():
            for k, kind, name in self.ref_order():
                if k not in sd:
                    if strict:
                        raise KeyError(f"adapter has no {k}")
                    continue
                v = sd[k]
                dst = self.to_ref(flat, name, kind)
                if tuple(v.shape) != tuple(dst.shape):
                    raise ValueError(f"{k}: adapter {tuple(v.shape)} vs engine {tuple(dst.shape)}")
                dst.copy_(v.to(dst.device, dst.dtype))
                n += 1
        if strict and n != 2 * len(self.members):
            raise ValueError(f"loaded {n} of {2 * len(self.members)} adapter tensors")
        return n

    def load_binned(self, sd: dict, strict: bool = True) -> int:
        n = self.load_into(self.master, sd, strict)
        self.push()
        return n

    # ---------------------------------------------------- PEFT format (one adapter, K == 1)
    @staticmethod
    def peft_key(name: str, kind: str) -> str:
        return f"base_model.model.{name}.lora_{kind}.weight"

    def state_peft(self) -> dict:
        """PEFT LoRA tensors (lora_A [r, in], lora_B [out, r]) for peft.PeftModel on the upstream transformer."""
        if self.K != 1:
            raise ValueError("PEFT format holds one adapter; n_bins > 1 saves with state_binned()")
        return {self.peft_key(n, kind): self.to_ref(self.master, n, kind)[0].detach().to("cpu", torch.float32)
                .contiguous() for _, kind, n in self.ref_order()}

    def load_peft(self, sd: dict) -> int:
        if self.K != 1:
            raise ValueError("a PEFT adapter holds one LoRA; this engine has n_bins > 1")
        binned = {}
        for k, kind, n in self.ref_order():
            pk = self.peft_key(n, kind)
            if pk not in sd:
                raise KeyError(f"adapter has no {pk}")
            binned[k] = sd[pk].unsqueeze(0)
        return self.load_binned(binned)


def acc_mm(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0, state: dict = {"ok": None}):
    """acc += alpha * (a @ b), acc float32 and a, b bf16: one cuBLAS GEMM accumulating in fp32 into acc."""
    if acc.is_cuda and a.dtype in (torch.bfloat16, torch.float16) and acc.dtype == torch.float32 \
            and state["ok"] is not False:
        try:
            torch.addmm(acc, a, b, torch.float32, alpha=alpha, out=acc)
            state["ok"] = True
            return
        except (RuntimeError, TypeError):
            state["ok"] = False
    acc.add_(a @ b, alpha=alpha) if a.dtype == acc.dtype else acc.add_((a @ b).to(acc.dtype), alpha=alpha)


# =========================================================================== packed batch
@dataclass
class Packed:
    B: int
    n_img: int
    lens: list                       # text tokens per row
    T: int
    cu_cpu: list
    max_len: int
    cu: torch.Tensor                 # int32 [B + 1] (device)
    seg: torch.Tensor                # long [T]: sample of each token
    text_idx: torch.Tensor           # long [sum lens]: packed positions of text tokens (row order)
    img_idx: torch.Tensor            # long [B * n_img]
    ind: torch.Tensor                # long [T]: 1 for image tokens, 0 for text (the indicator embedding index)
    pos: torch.Tensor                # long [T, 3] position ids
    seg_img: torch.Tensor            # long [B * n_img]
    flat_text_src: torch.Tensor      # long [sum lens]: row-major index of each text token in a padded (B, L) layout
    L_pad: int
    mt: int
    text_src_prefix: torch.Tensor | None = None   # long [sum lens]: index of each text token in a (B, mt) prefix layout


def pack_layout(lens: list, grid_h: int, grid_w: int, device, image_offset: int = IMAGE_POSITION_OFFSET,
                pin: bool = True) -> Packed:
    """CPU-built packed layout of a batch laid out like the upstream pipeline's inputs ([pad][text][image] per row,
    text positions 0..n-1 on all three axes, image positions (0, h, w) + IMAGE_POSITION_OFFSET), without the pad."""
    B = len(lens)
    n_img = grid_h * grid_w
    mt = max(lens)
    L = mt + n_img
    seg, text_idx, img_idx, ind, pos, flat_src, pre_src = [], [], [], [], [], [], []
    cu = [0]
    hh = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    ww = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    img_pos = torch.stack([torch.zeros_like(hh), hh, ww], 1) + image_offset
    for b, n in enumerate(lens):
        o = cu[-1]
        seg.append(torch.full((n + n_img,), b, dtype=torch.long))
        text_idx.append(torch.arange(o, o + n))
        img_idx.append(torch.arange(o + n, o + n + n_img))
        ind.append(torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n_img, dtype=torch.long)]))
        tp = torch.arange(n)
        pos.append(torch.cat([torch.stack([tp, tp, tp], 1), img_pos]))
        flat_src.append(b * L + (mt - n) + torch.arange(n))
        pre_src.append(b * mt + (mt - n) + torch.arange(n))
        cu.append(o + n + n_img)
    T = cu[-1]

    def dev(x):
        x = x.contiguous()
        if pin and torch.device(device).type == "cuda":
            return x.pin_memory().to(device, non_blocking=True)
        return x.to(device)
    seg_t = torch.cat(seg)
    img_t = torch.cat(img_idx)
    return Packed(B=B, n_img=n_img, lens=list(lens), T=T, cu_cpu=cu, max_len=max(n + n_img for n in lens),
                  cu=dev(torch.tensor(cu, dtype=torch.int32)), seg=dev(seg_t), text_idx=dev(torch.cat(text_idx)),
                  img_idx=dev(img_t), ind=dev(torch.cat(ind)), pos=dev(torch.cat(pos)),
                  seg_img=dev(seg_t[img_t]), flat_text_src=dev(torch.cat(flat_src)), L_pad=L, mt=mt,
                  text_src_prefix=dev(torch.cat(pre_src)))


# =========================================================================== attention
_FA3: dict = {}


def _fa3():
    if "m" not in _FA3:
        try:
            import flash_attn_interface as m
            _FA3["m"] = m
        except Exception:               # noqa: BLE001
            _FA3["m"] = None
    return _FA3["m"]


def attn_fwd(q, k, v, p: Packed, impl: str):
    """q, k, v [T, H, hd] -> (out [T, H, hd], lse or None). Bidirectional within each sample."""
    sc = q.shape[-1] ** -0.5
    if impl == "fa3":
        m = _fa3()
        o, lse, *_ = m._flash_attn_forward(q, k, v, cu_seqlens_q=p.cu, cu_seqlens_k=p.cu, max_seqlen_q=p.max_len,
                                            max_seqlen_k=p.max_len, softmax_scale=sc, causal=False)
        return o, lse
    out = torch.empty_like(q)
    for a, b in zip(p.cu_cpu[:-1], p.cu_cpu[1:]):
        out[a:b] = F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1),
                                                  v[a:b].transpose(0, 1)).transpose(0, 1)
    return out, None


def attn_bwd(do, q, k, v, o, lse, p: Packed, impl: str):
    sc = q.shape[-1] ** -0.5
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if impl == "fa3":
        m = _fa3()
        m._flash_attn_backward(do.contiguous(), q, k, v, o, lse, p.cu, p.cu, None, None, p.max_len, p.max_len,
                               dq, dk, dv, softmax_scale=sc, is_causal=False)
        return dq, dk, dv
    for a, b in zip(p.cu_cpu[:-1], p.cu_cpu[1:]):
        with torch.enable_grad():
            qq, kk, vv = (t[a:b].detach().transpose(0, 1).requires_grad_(True) for t in (q, k, v))
            oo = F.scaled_dot_product_attention(qq, kk, vv)
            gq, gk, gv = torch.autograd.grad(oo, (qq, kk, vv), do[a:b].transpose(0, 1))
        dq[a:b], dk[a:b], dv[a:b] = gq.transpose(0, 1), gk.transpose(0, 1), gv.transpose(0, 1)
    return dq, dk, dv


# =========================================================================== the model
@dataclass
class Block:
    groups: dict                     # kind -> Group
    w_an1: torch.Tensor
    w_an2: torch.Tensor
    w_fn1: torch.Tensor
    w_fn2: torch.Tensor
    w_nq: torch.Tensor
    w_nk: torch.Tensor
    adaln: nn.Module                 # frozen adaLN modulation Linear (compute dtype)
    eps: float
    eps_qk: float


@dataclass
class Ctx:
    p: Packed
    mods: list                       # per block (s_msa, g_msa, s_mlp, g_mlp), each [B, D]
    cos: torch.Tensor
    sin: torch.Tensor
    wtok: torch.Tensor               # [T, K] bin weights per token (compute dtype)
    scale_f: torch.Tensor            # [B, D] final-layer scale


def bin_blend_weights(t: torch.Tensor, n_bins: int, base_bins: int) -> torch.Tensor:
    """(B,) t (t = 0 noise, t = 1 data) -> (B, n_bins - base_bins) adapter weights. The timeline is cut into n_bins
    equal bins, adapter k is centred on its bin and the weights blend linearly between neighbouring centres (a
    partition of unity); the last `base_bins` bins (lowest noise) have no adapter, faded out over the half-bin
    before them. n_bins = 1, base_bins = 0 gives weight 1 everywhere: one ordinary LoRA."""
    k_a = n_bins - base_bins
    t = t.float().reshape(-1)
    pos = (t * n_bins - 0.5).clamp(0, k_a - 1)
    k0 = pos.floor().long().clamp(max=k_a - 1)
    frac = (pos - k0.float()).clamp(0, 1)
    w = torch.zeros(t.shape[0], k_a, device=t.device)
    w.scatter_(1, k0[:, None], (1 - frac)[:, None])
    k1 = (k0 + 1).clamp(max=k_a - 1)
    w.scatter_add_(1, k1[:, None], torch.where(k1 > k0, frac, torch.zeros_like(frac))[:, None])
    if base_bins == 0:
        return w
    t_base = 1.0 - base_bins / n_bins
    fade = ((t_base - t) * 2 * n_bins).clamp(0, 1)
    return w * fade[:, None]


class FastDiT:
    """Owns the frozen DiT weights (taken out of an Ideogram4Transformer) and the LoRA store.

    tr: an Ideogram4Transformer (the ideogram4 package's, or perf_kit/dit/ideogram4/upstream's); its five target
    Linears per block may be the package's Fp8Linear. With release=True the target modules are removed from `tr`
    (their weights now live in the engine)."""

    def __init__(self, tr: nn.Module, *, rank: int, alpha: float, rslora: bool = True, n_bins: int = 1,
                 base_bins: int = 0, compute_dtype=torch.bfloat16, master_dtype=torch.float32, storage: str = "fp8",
                 attn: str = "auto", fp8_forward: bool = False, release: bool = True):
        if not 0 <= base_bins < n_bins:
            raise ValueError(f"base_bins must be in [0, n_bins - 1], got {base_bins} of {n_bins}")
        self.tr = tr
        cfg = tr.config
        self.cfg = cfg
        self.cdt = compute_dtype
        self.rdt = torch.float64 if compute_dtype == torch.float64 else torch.float32      # residual dtype
        self.D, self.H, self.F = cfg.emb_dim, cfg.num_heads, cfg.intermediate_size
        self.hd = self.D // self.H
        self.n_bins, self.base_bins = n_bins, base_bins
        self.K = n_bins - base_bins
        self.rank, self.alpha, self.rslora = rank, alpha, rslora
        self.scale = alpha / math.sqrt(rank) if rslora else alpha / rank
        dev = tr.input_proj.weight.device
        self.device = dev
        self.scratch = Scratch()
        self.blocks: list[Block] = []
        groups = []
        for i, blk in enumerate(tr.layers):
            mod = {n: blk.get_submodule(n) for n in TARGETS}
            gd = {}
            for kind, names in GROUPS:
                fz = frozen_from([mod[n] for n in names], compute_dtype, storage)
                mem, off = [], 0
                for n in names:
                    o = mod[n].weight.shape[0]
                    mem.append(Member(f"layers.{i}.{n}", o, off))
                    off += o
                g = Group(i, kind, fz, mem, fz.inp, fp8_fwd=fp8_forward and fz.codes is not None)
                gd[kind] = g
                groups.append(g)
            att = blk.attention
            f32 = lambda t: t.detach().to(self.rdt).clone()  # noqa: E731
            self.blocks.append(Block(gd, f32(blk.attention_norm1.weight), f32(blk.attention_norm2.weight),
                                     f32(blk.ffn_norm1.weight), f32(blk.ffn_norm2.weight),
                                     f32(att.norm_q.weight), f32(att.norm_k.weight), blk.adaln_modulation,
                                     blk.attention_norm1.eps, att.norm_q.eps))
            if release:
                for n in TARGETS:
                    parent, leaf = n.split(".")
                    setattr(getattr(blk, parent), leaf, None)
        self.lora = LoraStore(groups, self.K, rank, self.scale, dev, master_dtype, compute_dtype)
        if attn == "auto":
            attn = "fa3" if (dev.type == "cuda" and _fa3() is not None and compute_dtype == torch.bfloat16) else "ref"
        self.attn = attn
        self.n_layers = len(self.blocks)

    # ------------------------------------------------------------------ parameter count / memory
    def frozen_bytes(self) -> int:
        return sum(g.frozen.nbytes() for b in self.blocks for g in b.groups.values())

    def saved_bytes_per_token(self) -> int:
        """Activation bytes kept per token for one fully saved block (excludes the float32 residual input)."""
        e = torch.tensor([], dtype=self.cdt).element_size()
        kr = self.K * self.rank
        return e * (3 * self.D + self.D + self.D + 2 * self.F + self.D + 7 * kr) + 4 * self.H

    # ------------------------------------------------------------------ frozen, gradient-free preparation
    @torch.no_grad()
    def prepare(self, p: Packed, llm_text: torch.Tensor, z_img: torch.Tensor, t: torch.Tensor) -> tuple:
        """llm_text [sum lens, Dllm] (text tokens, row order), z_img [B * n_img, C] noisy latents, t [B] float32.
        Returns (x0 residual [T, D], Ctx). Mirrors Ideogram4Transformer.forward up to the first block."""
        tr, cdt = self.tr, self.cdt
        h = torch.empty(p.T, self.D, dtype=cdt, device=self.device)
        nrm = tr.llm_cond_norm
        fn = ops.rmsnorm_w(llm_text.to(cdt), nrm.weight, nrm.eps, cdt)
        h[p.text_idx] = tr.llm_cond_proj(fn)
        h[p.img_idx] = tr.input_proj(z_img.to(cdt))
        h = h + tr.embed_image_indicator.weight[p.ind]
        x0 = h.to(self.rdt)
        tc = t.to(cdt)
        t_cond = tr.t_embedding(tc)                                    # [B, D]
        ada = F.silu(tr.adaln_proj(t_cond))                            # [B, adaln_dim]
        mods = []
        for blk in self.blocks:
            m = blk.adaln(ada)
            s1, g1, s2, g2 = m.chunk(4, dim=-1)
            mods.append((1.0 + s1, torch.tanh(g1), 1.0 + s2, torch.tanh(g2)))
        scale_f = 1.0 + tr.final_layer.adaln_modulation(F.silu(ada))
        cos, sin = tr.rotary_emb(p.pos.unsqueeze(0))                   # float32 tables (see stack.restore_rope_precision)
        wt = bin_blend_weights(t, self.n_bins, self.base_bins).to(self.device)
        return x0, Ctx(p, mods, cos[0].to(cdt), sin[0].to(cdt), wt[p.seg].to(cdt), scale_f)

    # ------------------------------------------------------------------ fused projection + binned LoRA
    def _gfwd(self, g: Group, x: torch.Tensor, ctx: Ctx):
        T = x.shape[0]
        if g.fp8_fwd:
            y = fp8_linear(x, g.frozen.codes, g.frozen.scale_row, self.cdt)
        else:
            y = x @ g.frozen.weight(self.scratch, self.cdt).t()
        M, K, r = len(g.members), self.K, self.rank
        xa = x @ g.a_c.t()                                             # [T, M K r]
        xw = (xa.view(T, M, K, r) * ctx.wtok.view(T, 1, K, 1)).view(T, M * K * r)
        kr = K * r
        if M == 1:
            y.addmm_(xw, g.members[0].sb_c.t())
        else:
            for j, m in enumerate(g.members):
                y[:, m.off: m.off + m.out].addmm_(xw[:, j * kr: (j + 1) * kr], m.sb_c.t())
        return y, xw

    def _gbwd(self, g: Group, dy: torch.Tensor, x: torch.Tensor, xw: torch.Tensor, ctx: Ctx, need_dx: bool = True):
        T = dy.shape[0]
        M, K, r = len(g.members), self.K, self.rank
        kr = K * r
        if M == 1:
            dxw = dy @ g.members[0].sb_c
        else:
            dxw = torch.cat([dy[:, m.off: m.off + m.out] @ m.sb_c for m in g.members], 1)
        dxa = (dxw.view(T, M, K, r) * ctx.wtok.view(T, 1, K, 1)).view(T, M * kr)
        acc_mm(g.a_grad, dxa.t(), x)
        for j, m in enumerate(g.members):
            dym = dy if M == 1 else dy[:, m.off: m.off + m.out]
            acc_mm(m.b_grad, dym.t(), xw[:, j * kr: (j + 1) * kr], self.scale)
        if not need_dx:
            return None
        dx = dy @ g.frozen.weight(self.scratch, self.cdt)
        dx.addmm_(dxa, g.a_c)
        return dx

    # ------------------------------------------------------------------ one block
    def block_fwd(self, i: int, x: torch.Tensor, ctx: Ctx, save: bool):
        B = self.blocks[i]
        G = B.groups
        p, cdt = ctx.p, self.cdt
        s_msa, g_msa, s_mlp, g_mlp = ctx.mods[i]
        T = x.shape[0]
        a_in = ops.norm_mod(x, B.w_an1, s_msa, p.seg, B.eps, cdt)
        qkv, xw_qkv = self._gfwd(G["qkv"], a_in, ctx)
        del a_in
        q, k, v = ops.qk_norm_rope(qkv, B.w_nq, B.w_nk, ctx.cos, ctx.sin, self.H, B.eps_qk)
        o, lse = attn_fwd(q, k, v, p, self.attn)
        del q, k, v
        ao, xw_o = self._gfwd(G["o"], o.view(T, self.D), ctx)
        x1 = ops.gate_res(x, ao, B.w_an2, g_msa, p.seg, B.eps)
        f_in = ops.norm_mod(x1, B.w_fn1, s_mlp, p.seg, B.eps, cdt)
        g13, xw_13 = self._gfwd(G["w13"], f_in, ctx)
        del f_in
        hff = ops.swiglu(g13, self.F)
        fo, xw_2 = self._gfwd(G["w2"], hff, ctx)
        del hff
        x2 = ops.gate_res(x1, fo, B.w_fn2, g_mlp, p.seg, B.eps)
        saved = dict(qkv=qkv, o=o, lse=lse, ao=ao, g13=g13, fo=fo, xw=(xw_qkv, xw_o, xw_13, xw_2)) if save else None
        return x2, saved

    def block_bwd(self, i: int, x: torch.Tensor, s: dict, dx2: torch.Tensor, ctx: Ctx, need_dx: bool = True):
        B = self.blocks[i]
        G = B.groups
        p, cdt = ctx.p, self.cdt
        s_msa, g_msa, s_mlp, g_mlp = ctx.mods[i]
        T = x.shape[0]
        xw_qkv, xw_o, xw_13, xw_2 = s["xw"]
        # ---- MLP half: x2 = x1 + g_mlp * rmsnorm(fo) * w_fn2
        dfo = ops.gate_res_bwd(s["fo"], B.w_fn2, g_mlp, p.seg, dx2, B.eps)
        hff = ops.swiglu(s["g13"], self.F)
        dh = self._gbwd(G["w2"], dfo, hff, xw_2, ctx)
        del dfo, hff
        dg13 = ops.swiglu_bwd(s["g13"], dh, self.F)
        del dh
        x1 = ops.gate_res(x, s["ao"], B.w_an2, g_msa, p.seg, B.eps)
        f_in = ops.norm_mod(x1, B.w_fn1, s_mlp, p.seg, B.eps, cdt)
        df = self._gbwd(G["w13"], dg13, f_in, xw_13, ctx)
        del dg13, f_in
        dx1 = dx2 + ops.norm_mod_bwd(x1, B.w_fn1, s_mlp, p.seg, df, B.eps)
        del df, x1
        # ---- attention half: x1 = x + g_msa * rmsnorm(ao) * w_an2
        dao = ops.gate_res_bwd(s["ao"], B.w_an2, g_msa, p.seg, dx1, B.eps)
        do = self._gbwd(G["o"], dao, s["o"].view(T, self.D), xw_o, ctx)
        del dao
        q, k, v = ops.qk_norm_rope(s["qkv"], B.w_nq, B.w_nk, ctx.cos, ctx.sin, self.H, B.eps_qk)
        dq, dk, dv = attn_bwd(do.view(T, self.H, self.hd), q, k, v, s["o"], s["lse"], p, self.attn)
        del q, k, v, do
        dqkv = ops.qk_norm_rope_bwd(s["qkv"], B.w_nq, B.w_nk, ctx.cos, ctx.sin, dq, dk, dv, self.H, B.eps_qk)
        del dq, dk, dv
        a_in = ops.norm_mod(x, B.w_an1, s_msa, p.seg, B.eps, cdt)
        da = self._gbwd(G["qkv"], dqkv, a_in, xw_qkv, ctx, need_dx=need_dx)
        del dqkv, a_in
        if not need_dx:
            return None
        return dx1 + ops.norm_mod_bwd(x, B.w_an1, s_msa, p.seg, da, B.eps)

    # ------------------------------------------------------------------ final layer (image tokens only) + loss
    def final_fwd(self, x_img: torch.Tensor, ctx: Ctx) -> torch.Tensor:
        fl = self.tr.final_layer
        fin = ops.final_norm(x_img, ctx.scale_f, ctx.p.seg_img, fl.norm_final.eps, self.cdt)
        return fl.linear(fin)

    def loss_and_grad(self, x_last, ctx: Ctx, v_tgt, loss_scale: float):
        """Flow MSE over image tokens: mean over channels, mean over each sample's image tokens, mean over samples.
        Returns (loss [], per-sample loss [B], dx_last)."""
        p = ctx.p
        x_img = x_last[p.img_idx]
        fl = self.tr.final_layer
        fin = ops.final_norm(x_img, ctx.scale_f, p.seg_img, fl.norm_final.eps, self.cdt)
        pred = fl.linear(fin)
        err = pred.to(self.rdt) - v_tgt.to(self.rdt)                  # [B n_img, C]
        C = err.shape[-1]
        per_tok = err.pow(2).mean(-1)
        per_sample = per_tok.view(p.B, p.n_img).mean(1)
        loss = per_sample.mean()
        dpred = (err * (2.0 * loss_scale / (C * p.n_img * p.B))).to(self.cdt)
        dfin = dpred @ fl.linear.weight.to(self.cdt)
        dx_img = ops.final_norm_bwd(x_img, ctx.scale_f, p.seg_img, dfin, fl.norm_final.eps)
        dx = torch.zeros_like(x_last)
        dx[p.img_idx] = dx_img.to(dx.dtype)
        return loss.detach(), per_sample.detach(), dx

    # ------------------------------------------------------------------ one micro-batch: forward + backward
    def train_micro(self, p: Packed, llm_text, z_img, v_tgt, t, loss_scale: float, n_save: int | None = None):
        """Accumulates loss_scale * d(loss)/d(LoRA) into lora.grad; returns (loss, per-sample loss)."""
        x, ctx = self.prepare(p, llm_text, z_img, t)
        n = self.n_layers if n_save is None else max(0, min(self.n_layers, n_save))
        xs, saved = [x], []
        for i in range(self.n_layers):
            x, s = self.block_fwd(i, x, ctx, save=i < n)
            xs.append(x)
            saved.append(s)
        loss, per_sample, dx = self.loss_and_grad(xs[-1], ctx, v_tgt, loss_scale)
        xs[-1] = None
        for i in reversed(range(self.n_layers)):
            s = saved[i]
            if s is None:                                              # recompute this block's activations
                _, s = self.block_fwd(i, xs[i], ctx, save=True)
            dx = self.block_bwd(i, xs[i], s, dx, ctx, need_dx=i > 0)
            saved[i] = xs[i] = s = None
        return loss, per_sample

    # ------------------------------------------------------------------ inference (sampler, validation)
    @torch.no_grad()
    def forward_padded(self, llm_features, x, t, position_ids, segment_ids, indicator) -> torch.Tensor:
        """Same signature and output as Ideogram4Transformer.forward (velocity, float32, meaningful at image
        positions only). Builds the packed layout from the indicator (one host sync; inference only)."""
        Bsz, L, C = x.shape
        ind = indicator.long()
        is_txt = ind == LLM_TOKEN_INDICATOR
        is_img = ind == OUTPUT_IMAGE_INDICATOR
        real = is_txt | is_img
        flat_real = real.reshape(-1).nonzero().squeeze(1)
        n_per = real.sum(1).tolist()
        cu = [0]
        for n in n_per:
            cu.append(cu[-1] + n)
        T = cu[-1]
        seg = torch.repeat_interleave(torch.arange(Bsz, device=x.device), real.sum(1))
        tx = is_txt.reshape(-1)[flat_real]
        im = is_img.reshape(-1)[flat_real]
        text_idx = tx.nonzero().squeeze(1)
        img_idx = im.nonzero().squeeze(1)
        n_img_per = is_img.sum(1).tolist()
        if len(set(n_img_per)) != 1:
            raise ValueError("forward_padded: rows with different image token counts")
        n_img = n_img_per[0]
        lens = is_txt.sum(1).tolist()
        p = Packed(B=Bsz, n_img=n_img, lens=lens, T=T, cu_cpu=cu, max_len=max(n_per),
                   cu=torch.tensor(cu, dtype=torch.int32, device=x.device), seg=seg, text_idx=text_idx,
                   img_idx=img_idx, ind=im.long(), pos=position_ids.reshape(-1, 3)[flat_real].long(),
                   seg_img=seg[img_idx], flat_text_src=flat_real[text_idx], L_pad=L, mt=max(lens) if lens else 0)
        tt = t.reshape(-1).float()
        if tt.numel() == 1 and Bsz > 1:
            tt = tt.expand(Bsz)
        llm_text = llm_features.reshape(Bsz * L, -1)[p.flat_text_src]
        z_img = x.reshape(Bsz * L, C)[flat_real[img_idx]]
        with torch.autocast(x.device.type, enabled=False):
            h, ctx = self.prepare(p, llm_text, z_img, tt)
            for i in range(self.n_layers):
                h, _ = self.block_fwd(i, h, ctx, save=False)
            pred = self.final_fwd(h[p.img_idx], ctx)
        out = torch.zeros(Bsz, L, C, dtype=torch.float32, device=x.device)
        out.view(Bsz * L, C)[flat_real[img_idx]] = pred.float()
        return out


class EngineModule(nn.Module):
    """nn.Module face of the engine for code that calls a transformer, e.g. the ideogram4 package's sampler:
    `pipe.conditional_transformer = EngineModule(engine)` renders with the current adapter."""

    def __init__(self, eng: FastDiT):
        super().__init__()
        self.eng = eng
        self.config = eng.cfg
        self._dev = eng.device

    @property
    def device(self):
        return self._dev

    def forward(self, *, llm_features, x, t, position_ids, segment_ids, indicator):
        return self.eng.forward_padded(llm_features, x, t, position_ids, segment_ids, indicator)
