"""Where frozen weights come from: a Hugging Face safetensors checkpoint (read tensor by tensor, straight to the GPU,
FP8 codes kept as codes) or an in-memory state dict (tests). No model object is built, so a 27B checkpoint is never
materialised twice."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch


class DictSource:
    def __init__(self, sd: dict):
        self.sd = sd

    def names(self):
        return set(self.sd)

    def get(self, name, device):
        return self.sd[name].to(device)

    def is_fp8(self, name) -> bool:
        return self.sd[name].dtype == torch.float8_e4m3fn and (name + "_scale_inv") in self.sd


class SafetensorsSource:
    def __init__(self, path: str):
        self.path = Path(path)
        ix = self.path / "model.safetensors.index.json"
        if ix.exists():
            self.map = json.loads(ix.read_text())["weight_map"]
        else:
            from safetensors import safe_open
            self.map = {}
            for f in sorted(self.path.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as sf:
                    for k in sf.keys():
                        self.map[k] = f.name
        self._open: dict = {}

    def names(self):
        return set(self.map)

    def _file(self, name, device):
        from safetensors import safe_open
        fn = self.map[name]
        key = (fn, str(device))
        if key not in self._open:
            self._open[key] = safe_open(str(self.path / fn), framework="pt", device=str(device))
        return self._open[key]

    def get(self, name, device):
        return self._file(name, device).get_tensor(name)

    def is_fp8(self, name) -> bool:
        if name + "_scale_inv" not in self.map:
            return False
        sl = self._file(name, "cpu").get_slice(name)
        return sl.get_dtype() in ("F8_E4M3", "float8_e4m3fn")

    def close(self):
        self._open.clear()


def load_adapter(path: str) -> tuple[dict, dict]:
    """(tensors, adapter_config) of a PEFT LoRA adapter dir."""
    from safetensors.torch import load_file
    p = Path(path)
    cfg = json.loads((p / "adapter_config.json").read_text())
    f = p / "adapter_model.safetensors"
    if f.exists():
        return load_file(str(f)), cfg
    return torch.load(p / "adapter_model.bin", map_location="cpu"), cfg


def save_adapter(path: str, tensors: dict, config: dict) -> None:
    """Atomic PEFT-format save (adapter_model.safetensors + adapter_config.json): readable by peft and vLLM."""
    from safetensors.torch import save_file
    p = Path(path)
    tmp = p.with_name("." + p.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp / "adapter_model.safetensors"),
              metadata={"format": "pt"})
    (tmp / "adapter_config.json").write_text(json.dumps(config, indent=2))
    if p.is_symlink():                    # e.g. final -> step-NNNNNN made by hand: drop the link only
        p.unlink()
    elif p.exists():
        os.replace(p, p.with_name(f".{p.name}.old.{int(time.time())}"))
    os.replace(tmp, p)
