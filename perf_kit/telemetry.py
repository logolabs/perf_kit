"""Where a step's GPU time goes: a torch.profiler summary by kernel family, GPU idle share and kernels per step."""
from __future__ import annotations

FAMILIES = (                  # order matters: "flash" contains "fla", so attention is matched first
    ("attention", ("flash", "fmha", "attention", "sdpa")),
    ("dequant", ("_dequant_kernel",)),
    ("gdn_fla", ("chunk_", "fused_recurrent", "gated_delta", "delta_rule", "l2norm", "fla::", "fla.", "solve_tril",
                 "wy_", "cumsum", "kkt", "tilelang", "tvm", "kernel_kernel")),
    ("gemm", ("gemm", "cutlass", "xmma", "nvjet", "cublas", "matmul", "sm90_", "sm80_")),
    ("optimizer", ("multi_tensor", "adam", "foreach")),
    ("norm", ("norm", "rms", "layer_norm", "_rms")),
    ("conv1d", ("conv1d", "causal_conv", "conv")),
    ("swiglu", ("_swiglu",)),
    ("copy_cast", ("copy", "cast", "to_copy", "fill", "memset", "cat")),
    ("reduce", ("reduce", "sum", "softmax", "cross_entropy", "nll", "log_softmax", "logsumexp")),
    ("elementwise", ("elementwise", "vectorized", "unrolled", "silu", "mul", "add", "sigmoid", "exp")),
)


def family(name: str) -> str:
    n = name.lower()
    for fam, keys in FAMILIES:
        if any(k in n for k in keys):
            return fam
    return "other"


def summarize(prof, steps: int, wall_s: float, top: int = 12) -> dict:
    """prof: a finished torch.profiler.profile over `steps` steps that took wall_s seconds (synchronised)."""
    ev = [e for e in prof.key_averages() if getattr(e, "device_type", None) is not None
          and e.device_type.name == "CUDA"]
    tot = sum(e.self_device_time_total for e in ev) / 1e6
    fam: dict = {}
    for e in ev:
        f = family(e.key)
        d = fam.setdefault(f, [0.0, 0])
        d[0] += e.self_device_time_total / 1e6
        d[1] += e.count
    cpu = [e for e in prof.key_averages() if getattr(e, "device_type", None) is not None
           and e.device_type.name == "CPU"]
    cpu_self = sum(e.self_cpu_time_total for e in cpu) / 1e6
    return {"steps": steps, "wall_s": round(wall_s, 3), "kernel_s": round(tot, 3),
            "gpu_idle": round(1 - tot / wall_s, 3) if wall_s else None,
            "kernels_per_step": sum(e.count for e in ev) // max(1, steps),
            "cpu_op_self_s_per_step": round(cpu_self / max(1, steps), 3),
            "families_s_per_step": {k: [round(v[0] / steps, 4), v[1] // steps] for k, v in
                                    sorted(fam.items(), key=lambda kv: -kv[1][0])},
            "top": [[e.key[:80], round(e.self_device_time_total / 1e6 / steps, 4), e.count // steps]
                    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:top]]}
