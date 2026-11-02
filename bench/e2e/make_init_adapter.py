"""Write one deterministic LoRA init adapter (PEFT format) that every benchmark arm loads, so all trainers start
from bit-identical LoRA weights: A ~ U(-1/sqrt(in), 1/sqrt(in)) (PEFT's default bound) from a per-module seed,
B = 0 (so step 1 sees the base model exactly). Module shapes come from the safetensors headers; no weights are read.

    python make_init_adapter.py --model <hf dir> --out <dir> [--rank 16] [--alpha 32] [--seed 0]

Targets: every projection of the text decoder's layers (Qwen3.5: Gated DeltaNet in_proj_qkv / in_proj_z /
out_proj, attention q/k/v/o, MLP gate/up/down); the vision tower and MTP head are not adapted.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

for _d in (Path(__file__).resolve().parent, Path(sys.argv[0]).resolve().parent):
    if (_d / "common.py").exists():
        sys.path.insert(0, str(_d))
        break
from common import TARGETS  # noqa: E402


def header_shapes(model_dir: Path) -> dict:
    shapes = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                shapes[k] = v["shape"]
    return shapes


def target_modules(shapes: dict, targets=TARGETS) -> dict:
    """{module name: (out, in)} for decoder-layer projections named in targets."""
    out = {}
    for k, shp in shapes.items():
        if not k.endswith(".weight") or ".layers." not in k or k.startswith("mtp.") or ".visual." in k:
            continue
        mod = k[: -len(".weight")]
        if mod.rsplit(".", 1)[-1] in targets and len(shp) == 2:
            out[mod] = (int(shp[0]), int(shp[1]))
    return dict(sorted(out.items()))


def main(argv=None) -> int:
    import torch
    from safetensors.torch import save_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    mods = target_modules(header_shapes(Path(a.model)))
    if not mods:
        print("no target modules found", file=sys.stderr)
        return 2
    tensors = {}
    for name, (o, i) in mods.items():
        g = torch.Generator().manual_seed(a.seed * 1_000_003 + zlib.crc32(name.encode()))
        bound = 1.0 / i ** 0.5
        tensors[f"base_model.model.{name}.lora_A.weight"] = torch.empty(a.rank, i).uniform_(-bound, bound, generator=g)
        tensors[f"base_model.model.{name}.lora_B.weight"] = torch.zeros(o, a.rank)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.alpha, "lora_dropout": 0.0,
           "bias": "none", "target_modules": sorted({m.rsplit(".", 1)[-1] for m in mods}), "fan_in_fan_out": False,
           "use_rslora": False, "use_dora": False, "inference_mode": False, "base_model_name_or_path": str(a.model)}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    n = sum(t.numel() for t in tensors.values())
    a_sum = float(sum(t.double().abs().sum() for k, t in tensors.items() if "lora_A" in k))
    (out / "init_meta.json").write_text(json.dumps({"modules": len(mods), "lora_params": n, "lora_A_abs_sum": a_sum,
                                                    "seed": a.seed}))
    print(json.dumps({"modules": len(mods), "lora_params": n, "lora_A_abs_sum": a_sum, "out": str(out)}))
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
