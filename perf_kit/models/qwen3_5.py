"""Fast training forward for the Qwen3.5 / 3.6 / 3.8 hybrid text decoder (Gated DeltaNet 3 : 1 gated attention), with
LoRA, on packed variable-length micro-batches. Same math as transformers' Qwen3_5 text model (tests compare the
reference kernels against it in float64), different execution:

  * projections sharing an input run as one GroupLinear (fused LoRA, LoRA grads straight into fp32 buffers);
  * frozen weights stay block-FP8 codes when the checkpoint is FP8 ("fp8" storage: dequantised exactly to bf16 per
    use; half the memory) or one bf16 copy ("bf16");
  * packed rows: flash-attn varlen, fla Gated DeltaNet with cu_seqlens and native GVA, fla's causal conv with
    cu_seqlens (transformers' torch conv fallback ignores row boundaries in padding-free batches);
  * Liger RMSNorm, a SwiGLU kernel reading the fused gate|up output, fla's fused gated RMSNorm;
  * per-layer activation checkpointing chosen per micro-batch (a bool per layer).

The model owns no nn.Module tree: it is built straight from a weight source (no 54 GB bf16 model in between).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..head import head_ce
from ..kernels import ops
from ..lora import GroupLinear, LoraStore, Member, Scratch
from ..packing import Packed

DEFAULT_TARGETS = ("in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")


@dataclass
class TextCfg:
    hidden: int
    inter: int
    layers: int
    layer_types: list
    nk: int
    nv: int
    dk: int
    dv: int
    conv_k: int
    nq: int
    nkv: int
    head_dim: int
    eps: float
    rope_theta: float
    rot_dim: int
    vocab: int
    tie: bool

    @classmethod
    def from_hf(cls, c) -> "TextCfg":
        c = getattr(c, "text_config", c)
        rp = getattr(c, "rope_parameters", None) or {}
        theta = rp.get("rope_theta", getattr(c, "rope_theta", 10000.0))
        partial = rp.get("partial_rotary_factor", getattr(c, "partial_rotary_factor", 1.0))
        hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        if rp.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rp.get('rope_type')}")
        if not getattr(c, "attn_output_gate", True):
            raise NotImplementedError("attention without output gate")
        return cls(c.hidden_size, c.intermediate_size, c.num_hidden_layers, list(c.layer_types),
                   c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim,
                   c.linear_conv_kernel_dim, c.num_attention_heads, c.num_key_value_heads, hd, c.rms_norm_eps,
                   float(theta), int(hd * partial), c.vocab_size, bool(getattr(c, "tie_word_embeddings", False)))


@dataclass
class Layer:
    kind: str                     # "linear_attention" | "full_attention"
    norm_in: torch.Tensor
    norm_post: torch.Tensor
    mix_in: GroupLinear
    mix_out: GroupLinear
    mlp_in: GroupLinear
    mlp_down: GroupLinear
    extra: dict = field(default_factory=dict)


class FastQwen35:
    def __init__(self, hf_config, source, *, prefix: str = "model.language_model.", head_name: str = "lm_head.weight",
                 targets=DEFAULT_TARGETS, rank: int = 128, alpha: float = 256, storage: str = "fp8", device="cuda",
                 dtype=torch.bfloat16, master_dtype=torch.float32, impl: str = "auto", fused_gate: bool = True,
                 fp8_forward: bool = False):
        self.cfg = c = TextCfg.from_hf(hf_config)
        self.fused_gate = fused_gate
        self.device, self.dtype, self.impl, self.rank = torch.device(device), dtype, impl, rank
        self.prefix, self.head_name, self.targets = prefix, head_name, set(targets)
        self.scaling = alpha / rank
        self.scratch = Scratch()
        dev = self.device
        get = lambda n: source.get(n, dev)                                       # noqa: E731

        def member(name: str, out: int) -> Member:
            m = Member(name=name, out=out, lora=name.rsplit(".", 1)[-1] in self.targets)
            wn = name + ".weight"
            if storage == "fp8" and source.is_fp8(wn):
                m.codes = get(wn).contiguous()
                m.scale_inv = get(wn + "_scale_inv").float().contiguous()
            elif source.is_fp8(wn):
                from ..kernels.dequant import dequant_torch
                m.w = dequant_torch(get(wn), get(wn + "_scale_inv"), dtype)
            else:
                m.w = get(wn).to(dtype)
            return m

        def group(members, inp) -> GroupLinear:
            st = "fp8" if any(m.codes is not None for m in members) else "bf16"
            return GroupLinear(members, inp, rank, self.scaling, st, self.scratch, dev, dtype)

        kd, vd = c.nk * c.dk, c.nv * c.dv
        self.layers: list[Layer] = []
        for i, kind in enumerate(c.layer_types[: c.layers]):
            p = f"{prefix}layers.{i}."
            mlp_in = group([member(p + "mlp.gate_proj", c.inter), member(p + "mlp.up_proj", c.inter)], c.hidden)
            mlp_down = group([member(p + "mlp.down_proj", c.hidden)], c.inter)
            if kind == "linear_attention":
                q = p + "linear_attn."
                mix_in = group([member(q + "in_proj_qkv", 2 * kd + vd), member(q + "in_proj_z", vd),
                                member(q + "in_proj_b", c.nv), member(q + "in_proj_a", c.nv)], c.hidden)
                mix_out = group([member(q + "out_proj", c.hidden)], vd)
                extra = {"conv": get(q + "conv1d.weight").to(dtype).reshape(2 * kd + vd, c.conv_k).contiguous(),
                         "A_log": get(q + "A_log"), "dt_bias": get(q + "dt_bias"),
                         "gnorm": get(q + "norm.weight").to(dtype)}
            elif kind == "full_attention":
                q = p + "self_attn."
                mix_in = group([member(q + "q_proj", 2 * c.nq * c.head_dim), member(q + "k_proj", c.nkv * c.head_dim),
                                member(q + "v_proj", c.nkv * c.head_dim)], c.hidden)
                mix_out = group([member(q + "o_proj", c.hidden)], c.nq * c.head_dim)
                extra = {"q_norm": get(q + "q_norm.weight").to(dtype), "k_norm": get(q + "k_norm.weight").to(dtype)}
            else:
                raise NotImplementedError(kind)
            self.layers.append(Layer(kind, get(p + "input_layernorm.weight").to(dtype),
                                     get(p + "post_attention_layernorm.weight").to(dtype),
                                     mix_in, mix_out, mlp_in, mlp_down, extra))
        self.embed = get(prefix + "embed_tokens.weight").to(dtype)
        self.norm = get(prefix + "norm.weight").to(dtype)
        self.head = self.embed if c.tie else get(head_name).to(dtype)
        self.groups = [g for L in self.layers for g in (L.mix_in, L.mix_out, L.mlp_in, L.mlp_down)]
        for g in self.groups:                     # lossy option: forward GEMMs on the FP8 codes (A/B-gated)
            g.fp8_fwd = bool(fp8_forward)
        self.lora = LoraStore(self.groups, rank, self.scaling, dev, compute_dtype=dtype, master_dtype=master_dtype)
        self.anchor = torch.zeros((), device=dev, dtype=dtype, requires_grad=True)
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, c.rot_dim, 2, dtype=torch.float) / c.rot_dim))
        self.inv_freq = inv.to(dev)

    # ------------------------------------------------------------------ accounting
    def frozen_linear_params(self) -> int:
        return sum(g.frozen_params() for g in self.groups)

    def head_params(self) -> int:
        return self.head.numel()

    def stored_bytes(self) -> dict:
        return {"frozen_linear": sum(g.stored_bytes() for g in self.groups),
                "embed_head": self.embed.numel() * 2 + (0 if self.cfg.tie else self.head.numel() * 2),
                "lora_store": self.lora.master.numel() * (2 * self.lora.master.element_size() + 2),
                "scratch": self.scratch.nbytes()}

    # ------------------------------------------------------------------ forward
    def rope(self, pos: torch.Tensor):
        f = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((f, f), -1)
        return emb.cos().to(self.dtype), emb.sin().to(self.dtype)

    @staticmethod
    def _apply_rope(x, cos, sin):
        rot = cos.shape[-1]
        c, s = cos[:, None, :], sin[:, None, :]
        xr, xp = x[..., :rot], x[..., rot:]
        half = rot // 2
        rh = torch.cat((-xr[..., half:], xr[..., :half]), -1)
        return torch.cat([xr * c + rh * s, xp], -1)

    def _mixer_gdn(self, L: Layer, x, packed: Packed):
        c, a = self.cfg, self.anchor
        kd, vd, T = c.nk * c.dk, c.nv * c.dv, x.shape[0]
        # torch.split, not y[:, a:b]: each slice's backward would zero-fill and copy a full-size gradient (plus the
        # adds that merge them); split's backward is one cat
        qkv, z, b, al = torch.split(L.mix_in(x, a), [2 * kd + vd, vd, c.nv, c.nv], dim=1)
        qkv = ops.conv_silu(qkv, L.extra["conv"], packed, self.impl)
        q, k, v = torch.split(qkv, [kd, kd, vd], dim=1)
        q, k, v = q.reshape(T, c.nk, c.dk), k.reshape(T, c.nk, c.dk), v.reshape(T, c.nv, c.dv)
        o = ops.gdn_gated(q, k, v, al, b, L.extra["A_log"], L.extra["dt_bias"], packed, self.impl, self.fused_gate)
        o = ops.gated_rmsnorm(o.reshape(-1, c.dv), z.reshape(-1, c.dv), L.extra["gnorm"], c.eps, self.impl)
        return L.mix_out(o.reshape(T, vd), a)

    def _mixer_attn(self, L: Layer, x, packed: Packed, cos, sin):
        c, a = self.cfg, self.anchor
        T, hd = x.shape[0], c.head_dim
        nq2 = 2 * c.nq * hd
        y = L.mix_in(x, a)
        qg, k, v = torch.split(y, [nq2, c.nkv * hd, c.nkv * hd], dim=1)
        q, gate = torch.split(qg.reshape(T, c.nq, 2 * hd), [hd, hd], dim=-1)
        gate = gate.reshape(T, c.nq * hd)
        k, v = k.reshape(T, c.nkv, hd), v.reshape(T, c.nkv, hd)
        q = ops.rmsnorm(q, L.extra["q_norm"], c.eps, self.impl)
        k = ops.rmsnorm(k, L.extra["k_norm"], c.eps, self.impl)
        q, k = self._apply_rope(q, cos, sin), self._apply_rope(k, cos, sin)
        o = ops.attention(q.contiguous(), k.contiguous(), v.contiguous(), packed, hd ** -0.5, self.impl)
        o = o.reshape(T, c.nq * hd) * torch.sigmoid(gate)
        return L.mix_out(o, a)

    def _layer(self, i: int, h, delta, packed: Packed, cos, sin):
        """(residual stream, pending residual update) -> the same after layer i. The residual add is fused into the
        following RMSNorm (one kernel each way), so a layer hands its MLP output to the next layer unadded."""
        L, c = self.layers[i], self.cfg
        if delta is None:
            x = ops.rmsnorm(h, L.norm_in, c.eps, self.impl)
        else:
            x, h = ops.add_rmsnorm(delta, h, L.norm_in, c.eps, "offset", self.impl)
        mix = (self._mixer_gdn(L, x, packed) if L.kind == "linear_attention"
               else self._mixer_attn(L, x, packed, cos, sin))
        x, h = ops.add_rmsnorm(mix, h, L.norm_post, c.eps, "offset", self.impl)
        act = ops.silu_mul(L.mlp_in(x, self.anchor), c.inter, self.impl)
        return h, L.mlp_down(act, self.anchor)

    def hidden(self, packed: Packed, ckpt=None):
        """Final normed hidden states [T, hidden] of a packed micro-batch; ckpt[i] True = recompute layer i."""
        h, delta = F.embedding(packed.ids, self.embed), None
        cos, sin = self.rope(packed.pos)
        for i in range(len(self.layers)):
            if ckpt is not None and ckpt[i] and torch.is_grad_enabled():
                h, delta = checkpoint(self._layer, i, h, delta, packed, cos, sin, use_reentrant=False,
                                      preserve_rng_state=False)
            else:
                h, delta = self._layer(i, h, delta, packed, cos, sin)
        return ops.add_rmsnorm(delta, h, self.norm, self.cfg.eps, "offset", self.impl)[0]

    def token_ce(self, packed: Packed, ckpt=None, chunk: int = 1024) -> torch.Tensor:
        """Per-target CE (fp32) at packed.keep, in that order."""
        h = self.hidden(packed, ckpt)
        return head_ce(h.index_select(0, packed.keep), self.head, packed.tgt.index_select(0, packed.keep), chunk)
