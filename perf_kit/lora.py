"""Frozen projections that share an input, computed as one block with fused LoRA, and a flat LoRA parameter store.

`GroupLinear` holds the frozen weights of several Linears that read the same input (q/k/v; gate/up; the four
Gated-DeltaNet input projections) as ONE [sum(out), in] bf16 matrix -- stored either as bf16, or as the
checkpoint's block-FP8 codes dequantised (exactly) into a shared scratch buffer right before each GEMM -- plus
LoRA on any subset of its members. Forward: y = x W^T (one GEMM), xa = x A_cat^T (one GEMM for all members' A),
y_i += xa_i (s B_i)^T (one addmm each). Backward: dx = dy W + dxa A_cat (GEMM + addmm), and the LoRA gradients
are accumulated straight into fp32 buffers (dA_cat = dxa^T x: one GEMM for all members; dB_i = s dy_i^T xa_i), so
LoRA weights are not autograd leaves: no AccumulateGrad nodes, no casts, no PEFT wrapper launches.

Numerics are those of PEFT under bf16 autocast with fp32 LoRA masters: bf16 GEMMs with fp32 accumulation, the
LoRA output added in the same addmm (one rounding instead of two), gradients rounded to bf16 per micro-batch and
accumulated in fp32, scaling s applied exactly (s = alpha / r is a power of two for the usual r / alpha).

`LoraStore` owns all adapters as two flat buffers ([all A | all B] fp32 masters + their grads, and bf16 compute
copies laid out the same), so the optimizer is ONE fused AdamW over one tensor, the gradient all-reduce is one
NCCL call, and refreshing the compute copies after a step is two kernels. Element-wise AdamW on the flat tensor is
the same update as per-tensor AdamW (no weight decay coupling across tensors; one shared step count).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .kernels.dequant import dequant_into


@dataclass
class Member:
    name: str                    # full module name of the Linear (e.g. model.language_model.layers.3.mlp.gate_proj)
    out: int
    off: int = 0                 # row offset in the group output
    lora: bool = False
    r_off: int = 0               # row offset in the group's A block
    # storage (exactly one of): bf16 weight, or fp8 codes + scale_inv
    w: torch.Tensor | None = None
    codes: torch.Tensor | None = None
    scale_inv: torch.Tensor | None = None
    bias: torch.Tensor | None = None         # frozen bias (Qwen2-style q/k/v), added in the same addmm
    # LoRA views (set by LoraStore.attach)
    a_master: torch.Tensor | None = None
    b_master: torch.Tensor | None = None
    a_grad: torch.Tensor | None = None
    b_grad: torch.Tensor | None = None
    a_bf16: torch.Tensor | None = None
    sb_bf16: torch.Tensor | None = None


class Scratch:
    """One bf16 buffer per weight shape, shared by all groups of that shape: FP8 groups dequantise into it right
    before each GEMM. Safe on one stream (each use is dequant -> GEMM, enqueued in order)."""

    def __init__(self):
        self.bufs: dict = {}

    def get(self, shape, device, dtype=torch.bfloat16):
        k = (tuple(shape), str(device), dtype)
        b = self.bufs.get(k)
        if b is None:
            b = self.bufs[k] = torch.empty(shape, dtype=dtype, device=device)
        return b

    def nbytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self.bufs.values())


class GroupLinear:
    def __init__(self, members: list[Member], in_features: int, rank: int, scaling: float, storage: str,
                 scratch: Scratch | None, device, dtype=torch.bfloat16):
        self.members, self.inp, self.rank, self.scaling = members, in_features, rank, scaling
        self.storage, self.scratch, self.device, self.dtype = storage, scratch, device, dtype
        off = r_off = 0
        for m in members:
            m.off, off = off, off + m.out
            if m.lora:
                m.r_off, r_off = r_off, r_off + rank
        self.out, self.r_total = off, r_off
        self.lora_members = [m for m in members if m.lora]
        self.fp8_fwd = False                   # set by the model: forward GEMMs in FP8 (lossy, A/B-gated option)
        self.bias = None
        if any(m.bias is not None for m in members):
            self.bias = torch.cat([m.bias.to(dtype) if m.bias is not None else torch.zeros(m.out, dtype=dtype,
                                   device=device) for m in members]).to(device)
            for m in members:
                m.bias = None
        self.w = None
        self.runs: list[dict] = []
        if storage == "bf16":                  # one persistent [out, in] matrix
            self.w = torch.empty(self.out, in_features, dtype=dtype, device=device)
            for m in members:
                self._fill(self.w[m.off: m.off + m.out], m)
                m.w = m.codes = m.scale_inv = None
        elif storage == "fp8":
            self._build_runs()
        else:
            raise ValueError(f"storage {storage!r}")
        # the A block's compute copy / master / grad are contiguous slices of the store (set in LoraStore.attach)
        self.a_bf16 = self.a_master = self.a_grad = None

    @staticmethod
    def _fill(dst, m: Member):
        if m.codes is not None:
            dequant_into(dst, m.codes, m.scale_inv)
        else:
            dst.copy_(m.w)

    def _build_runs(self):
        """Consecutive members of the same kind become ONE stored tensor (one dequant / copy launch per run): FP8
        members whose row counts are multiples of the 128-row scale block are concatenated with their scales;
        bf16 members are concatenated as they are."""
        runs, cur = [], None
        for m in self.members:
            kind = "fp8" if m.codes is not None else "bf16"
            if cur is not None and cur["kind"] == kind and (kind == "bf16" or cur["rows"] % 128 == 0):
                cur["m"].append(m)
                cur["rows"] += m.out
            else:
                cur = {"kind": kind, "off": m.off, "rows": m.out, "m": [m]}
                runs.append(cur)
        for r in runs:
            ms = r.pop("m")
            if r["kind"] == "fp8":
                r["codes"] = ms[0].codes if len(ms) == 1 else torch.cat([m.codes for m in ms])
                r["scale"] = ms[0].scale_inv if len(ms) == 1 else torch.cat([m.scale_inv for m in ms])
            else:
                r["w"] = ms[0].w if len(ms) == 1 else torch.cat([m.w for m in ms])
            for m in ms:
                m.codes = m.scale_inv = m.w = None
        self.runs = runs

    def weight(self) -> torch.Tensor:
        if self.w is not None:
            return self.w
        buf = self.scratch.get((self.out, self.inp), self.device, self.dtype)
        for r in self.runs:
            dst = buf[r["off"]: r["off"] + r["rows"]]
            if r["kind"] == "fp8":
                dequant_into(dst, r["codes"], r["scale"])
            else:
                dst.copy_(r["w"])
        return buf

    def frozen_params(self) -> int:
        return self.out * self.inp

    def stored_bytes(self) -> int:
        if self.w is not None:
            return self.w.numel() * self.w.element_size()
        return sum(r["codes"].numel() * r["codes"].element_size() + r["scale"].numel() * r["scale"].element_size()
                   if r["kind"] == "fp8" else r["w"].numel() * r["w"].element_size() for r in self.runs)

    def __call__(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """x [..., in] -> [..., out]; `anchor` is any tensor with requires_grad (makes the backward run even when
        x needs no gradient, e.g. right after the frozen embedding)."""
        shp = x.shape
        y = _GroupFn.apply(x.reshape(-1, self.inp), anchor, self)
        return y.view(*shp[:-1], self.out)


class _GroupFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, anchor, g: GroupLinear):
        if g.fp8_fwd and g.runs:
            y = _fp8_forward(g, x)
        else:
            w = g.weight()
            y = torch.addmm(g.bias, x, w.t()) if g.bias is not None else x @ w.t()
        xa = None
        if g.r_total:
            xa = x @ g.a_bf16.t()                                   # [T, R] for every member at once
            for m in g.lora_members:
                y[:, m.off: m.off + m.out].addmm_(xa[:, m.r_off: m.r_off + g.rank], m.sb_bf16.t())
        ctx.g = g
        ctx.save_for_backward(x, xa) if xa is not None else ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        g: GroupLinear = ctx.g
        saved = ctx.saved_tensors
        x = saved[0]
        dy = dy.contiguous()
        need_dx = ctx.needs_input_grad[0]
        dx = dy @ g.weight() if need_dx else None                   # FP8 storage: dequantised again (exact)
        if g.r_total:
            xa = saved[1]
            parts = [dy[:, m.off: m.off + m.out] @ m.sb_bf16 for m in g.lora_members]
            dxa = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
            if need_dx:
                dx.addmm_(dxa, g.a_bf16)
            acc_mm(g.a_grad, dxa.t(), x)                            # LoRA grads straight into the fp32 buffers
            for m in g.lora_members:
                acc_mm(m.b_grad, dy[:, m.off: m.off + m.out].t(), xa[:, m.r_off: m.r_off + g.rank], g.scaling)
        return dx, None, None


def _fp8_forward(g: "GroupLinear", x: torch.Tensor) -> torch.Tensor:
    """y = x W^T with the FP8 runs on the FP8 GEMM (activations quantised once per group) and bf16 runs in bf16."""
    from .kernels import fp8
    xq = fp8.quant_act(x) if (x.is_cuda and fp8.available()) else None
    parts = []
    for r in g.runs:
        if r["kind"] == "fp8":
            parts.append(fp8.fp8_gemm(x, r["codes"], r["scale"], xq))
        else:
            parts.append(x @ r["w"].t())
    y = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
    if g.bias is not None:
        y = y + g.bias
    return y


_FUSED_ACC = {"ok": None}


def acc_mm(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float = 1.0) -> None:
    """acc += alpha * (a @ b) with acc fp32 and a, b bf16. On CUDA one cuBLAS GEMM accumulating in fp32 straight
    into acc (torch.addmm out_dtype, Megatron's gradient-accumulation fusion: no bf16 rounding of the per-micro-batch
    gradient, one kernel instead of two); elsewhere / if unsupported: the GEMM in the inputs' dtype, then the add."""
    if acc.is_cuda and a.dtype in (torch.bfloat16, torch.float16) and acc.dtype == torch.float32             and _FUSED_ACC["ok"] is not False:
        try:
            torch.addmm(acc, a, b, torch.float32, alpha=alpha, out=acc)
            _FUSED_ACC["ok"] = True
            return
        except (RuntimeError, TypeError):
            _FUSED_ACC["ok"] = False
    acc.add_(a @ b, alpha=alpha)


@dataclass
class LoraStore:
    """All LoRA adapters of a model as flat buffers: master [A... | B...] fp32, its grad, bf16 compute copies with
    B pre-scaled by s. `slots` maps the PEFT tensor names to (kind, view) for load / save."""
    groups: list
    rank: int
    scaling: float
    device: object
    compute_dtype: torch.dtype = torch.bfloat16      # float64 in the CPU exactness tests
    master_dtype: torch.dtype = torch.float32
    master: torch.Tensor = field(init=False)
    grad: torch.Tensor = field(init=False)
    compute: torch.Tensor = field(init=False)
    n_a: int = field(init=False)
    slots: dict = field(init=False)

    def __post_init__(self):
        n_a = sum(g.r_total * g.inp for g in self.groups)
        n_b = sum(m.out * self.rank for g in self.groups for m in g.lora_members)
        self.n_a = n_a
        self.master = torch.zeros(n_a + n_b, dtype=self.master_dtype, device=self.device)
        self.grad = torch.zeros_like(self.master)
        self.compute = torch.zeros(n_a + n_b, dtype=self.compute_dtype, device=self.device)
        self.slots = {}
        oa, ob = 0, n_a
        for g in self.groups:
            if not g.r_total:
                continue
            n = g.r_total * g.inp
            g.a_master = self.master[oa: oa + n].view(g.r_total, g.inp)
            g.a_grad = self.grad[oa: oa + n].view(g.r_total, g.inp)
            g.a_bf16 = self.compute[oa: oa + n].view(g.r_total, g.inp)
            oa += n
            for m in g.lora_members:
                m.a_master = g.a_master[m.r_off: m.r_off + self.rank]
                m.a_grad = g.a_grad[m.r_off: m.r_off + self.rank]
                m.a_bf16 = g.a_bf16[m.r_off: m.r_off + self.rank]
                nb = m.out * self.rank
                m.b_master = self.master[ob: ob + nb].view(m.out, self.rank)
                m.b_grad = self.grad[ob: ob + nb].view(m.out, self.rank)
                m.sb_bf16 = self.compute[ob: ob + nb].view(m.out, self.rank)
                ob += nb
                self.slots[m.name] = m

    @property
    def numel(self) -> int:
        return self.master.numel()

    def init_new(self, seed: int = 0):
        """PEFT's default init: A ~ kaiming_uniform(a=sqrt(5)) (i.e. U(-1/sqrt(in), 1/sqrt(in))), B = 0."""
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for m in self.slots.values():
            bound = 1.0 / math.sqrt(m.a_master.shape[1])
            m.a_master.copy_(torch.empty(m.a_master.shape).uniform_(-bound, bound, generator=gen))
            m.b_master.zero_()
        self.push()

    def push(self):
        """Refresh the bf16 compute copies from the fp32 masters (after every optimizer step)."""
        self.compute[: self.n_a].copy_(self.master[: self.n_a])
        torch.mul(self.master[self.n_a:], self.scaling, out=self.compute[self.n_a:])   # fp32 product, one rounding

    def zero_grad(self):
        self.grad.zero_()

    # ------------------------------------------------------------- PEFT-format adapter I/O
    @staticmethod
    def peft_key(module_name: str, ab: str, prefix: str = "base_model.model.") -> str:
        return f"{prefix}{module_name}.lora_{ab}.weight"

    def load_peft(self, tensors: dict, prefix: str = "base_model.model.", strict: bool = True) -> int:
        n = 0
        with torch.no_grad():
            for name, m in self.slots.items():
                for ab, dst in (("A", m.a_master), ("B", m.b_master)):
                    k = self.peft_key(name, ab, prefix)
                    if k not in tensors:
                        if strict:
                            raise KeyError(f"adapter has no {k}")
                        continue
                    t = tensors[k]
                    if tuple(t.shape) != tuple(dst.shape):
                        raise ValueError(f"{k}: adapter {tuple(t.shape)} vs model {tuple(dst.shape)}")
                    dst.copy_(t.to(dst.device, dst.dtype))
                    n += 1
        if strict and n != 2 * len(self.slots):
            raise ValueError(f"loaded {n} of {2 * len(self.slots)} adapter tensors")
        self.push()
        return n

    def state_peft(self, prefix: str = "base_model.model.") -> dict:
        out = {}
        for name, m in self.slots.items():
            out[self.peft_key(name, "A", prefix)] = m.a_master.detach().to("cpu").clone()
            out[self.peft_key(name, "B", prefix)] = m.b_master.detach().to("cpu").clone()
        return out
