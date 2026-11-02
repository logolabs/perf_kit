"""Fast LoRA training forward for dense Llama-style decoders (Llama 2/3, Mistral, Qwen2/2.5, Qwen3): RMSNorm,
GQA softmax attention with RoPE (Qwen3: q/k RMSNorm; Qwen2: q/k/v bias), SwiGLU MLP. Same execution as
models/qwen3_5.py: q/k/v and gate/up as single GroupLinear GEMMs with fused LoRA, packed rows through flash-attn
varlen, Liger RMSNorm, the fused SwiGLU kernel, chunked head + CE, per-layer checkpoint plan. Built straight from
a weight source (bf16 or block-FP8 checkpoints). Sliding-window attention is not implemented (refused)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..head import head_ce
from ..kernels import ops
from ..lora import GroupLinear, LoraStore, Member, Scratch
from ..packing import Packed

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def rope_inv_freq(c, head_dim: int):
    """(inv_freq, attention_scaling) as transformers computes them for this config (default / llama3 / yarn / ...)."""
    rp = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
    kind = rp.get("rope_type", rp.get("type", "default")) if isinstance(rp, dict) else "default"
    partial = (rp.get("partial_rotary_factor") if isinstance(rp, dict) else None) or getattr(c, "partial_rotary_factor", 1.0)
    theta = (rp.get("rope_theta") if isinstance(rp, dict) else None) or getattr(c, "rope_theta", 10000.0)
    dim = int(head_dim * partial)
    if kind == "default":
        return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim)), 1.0
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    inv, scale = ROPE_INIT_FUNCTIONS[kind](c, "cpu")
    return inv.float(), float(scale)


@dataclass
class DenseCfg:
    hidden: int
    inter: int
    layers: int
    nq: int
    nkv: int
    head_dim: int
    eps: float
    vocab: int
    tie: bool
    qk_norm: bool
    bias: bool

    @classmethod
    def from_hf(cls, c, names) -> "DenseCfg":
        c = getattr(c, "text_config", c)
        if getattr(c, "sliding_window", None) and getattr(c, "use_sliding_window", True) and \
                getattr(c, "model_type", "") in ("mistral",):
            raise NotImplementedError("sliding-window attention")
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        qk = any(n.endswith("self_attn.q_norm.weight") for n in names)
        bias = any(n.endswith("self_attn.q_proj.bias") for n in names)
        return cls(c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.num_attention_heads,
                   c.num_key_value_heads, hd, c.rms_norm_eps, c.vocab_size, bool(getattr(c, "tie_word_embeddings", False)),
                   qk, bias)


class FastDense:
    def __init__(self, hf_config, source, *, prefix: str = "model.", head_name: str = "lm_head.weight",
                 targets=DEFAULT_TARGETS, rank: int = 16, alpha: float = 32, storage: str = "fp8", device="cuda",
                 dtype=torch.bfloat16, master_dtype=torch.float32, impl: str = "auto"):
        names = source.names()
        self.cfg = c = DenseCfg.from_hf(hf_config, names)
        self.device, self.dtype, self.impl, self.rank = torch.device(device), dtype, impl, rank
        self.prefix, self.targets, self.scaling = prefix, set(targets), alpha / rank
        self.scratch = Scratch()
        dev = self.device
        get = lambda n: source.get(n, dev)                                       # noqa: E731

        def member(name, out):
            m = Member(name=name, out=out, lora=name.rsplit(".", 1)[-1] in self.targets)
            wn = name + ".weight"
            if storage == "fp8" and source.is_fp8(wn):
                m.codes, m.scale_inv = get(wn).contiguous(), get(wn + "_scale_inv").float().contiguous()
            elif source.is_fp8(wn):
                from ..kernels.dequant import dequant_torch
                m.w = dequant_torch(get(wn), get(wn + "_scale_inv"), dtype)
            else:
                m.w = get(wn).to(dtype)
            if name + ".bias" in names:
                m.bias = get(name + ".bias")
            return m

        def group(ms, inp):
            st = "fp8" if any(m.codes is not None for m in ms) else "bf16"
            return GroupLinear(ms, inp, rank, self.scaling, st, self.scratch, dev, dtype)

        self.layers = []
        for i in range(c.layers):
            p = f"{prefix}layers.{i}."
            a = p + "self_attn."
            L = {"norm_in": get(p + "input_layernorm.weight").to(dtype),
                 "norm_post": get(p + "post_attention_layernorm.weight").to(dtype),
                 "qkv": group([member(a + "q_proj", c.nq * c.head_dim), member(a + "k_proj", c.nkv * c.head_dim),
                               member(a + "v_proj", c.nkv * c.head_dim)], c.hidden),
                 "o": group([member(a + "o_proj", c.hidden)], c.nq * c.head_dim),
                 "gu": group([member(p + "mlp.gate_proj", c.inter), member(p + "mlp.up_proj", c.inter)], c.hidden),
                 "down": group([member(p + "mlp.down_proj", c.hidden)], c.inter)}
            if c.qk_norm:
                L["q_norm"], L["k_norm"] = get(a + "q_norm.weight").to(dtype), get(a + "k_norm.weight").to(dtype)
            self.layers.append(L)
        self.embed = get(prefix + "embed_tokens.weight").to(dtype)
        self.norm = get(prefix + "norm.weight").to(dtype)
        self.head = self.embed if (c.tie or head_name not in names) else get(head_name).to(dtype)
        self.groups = [L[k] for L in self.layers for k in ("qkv", "o", "gu", "down")]
        self.lora = LoraStore(self.groups, rank, self.scaling, dev, compute_dtype=dtype, master_dtype=master_dtype)
        self.anchor = torch.zeros((), device=dev, dtype=dtype, requires_grad=True)
        inv, self.rope_scale = rope_inv_freq(getattr(hf_config, "text_config", hf_config), c.head_dim)
        self.inv_freq = inv.to(dev)

    def frozen_linear_params(self) -> int:
        return sum(g.frozen_params() for g in self.groups)

    def head_params(self) -> int:
        return self.head.numel()

    def stored_bytes(self) -> dict:
        return {"frozen_linear": sum(g.stored_bytes() for g in self.groups),
                "embed_head": self.embed.numel() * 2 + (0 if self.head is self.embed else self.head.numel() * 2),
                "lora_store": self.lora.master.numel() * (2 * self.lora.master.element_size() + 2),
                "scratch": self.scratch.nbytes()}

    def rope(self, pos):
        f = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), -1)
        return (emb.cos() * self.rope_scale).to(self.dtype), (emb.sin() * self.rope_scale).to(self.dtype)

    @staticmethod
    def _apply_rope(x, cos, sin):
        rot = cos.shape[-1]
        c, s = cos[:, None, :], sin[:, None, :]
        xr, xp = x[..., :rot], x[..., rot:]
        half = rot // 2
        rh = torch.cat((-xr[..., half:], xr[..., :half]), -1)
        out = xr * c + rh * s
        return out if xp.shape[-1] == 0 else torch.cat([out, xp], -1)

    def _layer(self, i, h, delta, packed: Packed, cos, sin):
        L, c, a = self.layers[i], self.cfg, self.anchor
        T, hd = h.shape[0], c.head_dim
        if delta is None:
            x = ops.rmsnorm_llama(h, L["norm_in"], c.eps, self.impl)
        else:
            x, h = ops.add_rmsnorm(delta, h, L["norm_in"], c.eps, "llama", self.impl)
        q, k, v = torch.split(L["qkv"](x, a), [c.nq * hd, c.nkv * hd, c.nkv * hd], dim=1)  # one cat in backward
        q, k, v = q.reshape(T, c.nq, hd), k.reshape(T, c.nkv, hd), v.reshape(T, c.nkv, hd)
        if c.qk_norm:
            q = ops.rmsnorm_llama(q, L["q_norm"], c.eps, self.impl)
            k = ops.rmsnorm_llama(k, L["k_norm"], c.eps, self.impl)
        q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
        o = ops.attention(q.contiguous(), k.contiguous(), v.contiguous(), packed, hd ** -0.5, self.impl)
        x, h = ops.add_rmsnorm(L["o"](o.reshape(T, c.nq * hd), a), h, L["norm_post"], c.eps, "llama", self.impl)
        return h, L["down"](ops.silu_mul(L["gu"](x, a), c.inter, self.impl), a)

    def hidden(self, packed: Packed, ckpt=None):
        h, delta = F.embedding(packed.ids, self.embed), None
        cos, sin = self.rope(packed.pos)
        for i in range(len(self.layers)):
            if ckpt is not None and ckpt[i] and torch.is_grad_enabled():
                h, delta = checkpoint(self._layer, i, h, delta, packed, cos, sin, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                h, delta = self._layer(i, h, delta, packed, cos, sin)
        return ops.add_rmsnorm(delta, h, self.norm, self.cfg.eps, "llama", self.impl)[0]

    def token_ce(self, packed: Packed, ckpt=None, chunk: int = 1024) -> torch.Tensor:
        h = self.hidden(packed, ckpt)
        return head_ce(h.index_select(0, packed.keep), self.head, packed.tgt.index_select(0, packed.keep), chunk)
