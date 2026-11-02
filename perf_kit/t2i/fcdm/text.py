"""Text encoders for the FCDM trainer (frozen by default), and compile_forward for the frozen one.

  ettin   jhu-clsp/ettin-encoder-68m (MIT; a ModernBERT encoder, hidden 512): Agate's encoder; an Agate release
          ships its fine-tuned copy in <release>/text_encoder
  t5      google/flan-t5-base (hidden 768)
  hash    deterministic per-byte embeddings, no download: tests and synthetic runs

compile_forward(module) compiles the module's forward only. Wrapping the module itself in torch.compile would put
`_orig_mod.` in front of every parameter name, so a checkpoint or a state-dict load stops matching (the trap that
failed the first pilot of this flag); compiling the bound forward keeps the module, and its names, as they are.
"""
from __future__ import annotations

import threading

import torch

# One lock for every call into a shared fast tokenizer: each call re-sets the Rust tokenizer's padding /
# truncation state, so two threads tokenizing at once can encode a batch with the other's settings.
TOK_LOCK = threading.Lock()


def _pad_ids(ids, mask, pad_to: int, pad_id: int):
    extra = (-ids.shape[1]) % pad_to
    if extra:
        ids = torch.nn.functional.pad(ids, (0, extra), value=pad_id)
        mask = torch.nn.functional.pad(mask, (0, extra))
    return ids, mask


class HFTextEncoder:
    """tokenize (CPU, thread-safe) -> encode (device, bf16 autocast on CUDA) -> (ctx (B, L, dim) fp32, mask (B, L))."""

    def __init__(self, model, tok, device, max_len: int):
        self.model, self.tok, self.device, self.max_len = model.to(device), tok, torch.device(device), max_len
        self.model.eval().requires_grad_(False)
        cfg = self.model.config
        self.dim = getattr(cfg, "hidden_size", None) or cfg.d_model

    def tokenize(self, captions: list[str], pad_to: int = 1):
        """`pad_to` rounds the length up (masked padding) so a compiled model sees few distinct shapes."""
        with TOK_LOCK:
            t = self.tok(captions, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt")
        return _pad_ids(t["input_ids"], t["attention_mask"], pad_to, self.tok.pad_token_id)

    @torch.no_grad()
    def encode(self, ids, mask):
        ids, mask = ids.to(self.device, non_blocking=True), mask.to(self.device, non_blocking=True)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        return h.float(), mask

    def __call__(self, captions: list[str]):
        return self.encode(*self.tokenize(captions))


def _load_modernbert(path: str):
    """AutoModel.from_pretrained, or -- where transformers refuses a .bin checkpoint on torch < 2.6 (Ettin ships only
    pytorch_model.bin) -- build from the config and load the state dict with weights_only=True."""
    from transformers import AutoConfig, AutoModel
    try:
        return AutoModel.from_pretrained(path, torch_dtype=torch.float32)
    except ValueError:
        import os
        m = AutoModel.from_config(AutoConfig.from_pretrained(path))
        f = os.path.join(path, "pytorch_model.bin") if os.path.isdir(path) else None
        if f is None:
            from huggingface_hub import hf_hub_download
            f = hf_hub_download(path, "pytorch_model.bin")
        sd = {k.removeprefix("model."): v for k, v in torch.load(f, map_location="cpu", weights_only=True).items()}
        missing, _ = m.load_state_dict(sd, strict=False)
        if missing:
            raise KeyError(f"{path}: encoder tensors missing from the checkpoint: {missing[:5]}")
        return m


class HashTextEncoder:
    """Deterministic per-byte embeddings with the same interface; "" gives one token (the CFG context)."""

    def __init__(self, dim: int = 32, device="cpu", max_len: int = 128, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(257, dim, generator=g)
        self.device, self.max_len, self.dim = torch.device(device), max_len, dim
        self.model = None

    def tokenize(self, captions: list[str], pad_to: int = 1):
        ids = [[256] + list(c.encode("utf-8"))[: self.max_len - 1] for c in captions]
        L = max(len(i) for i in ids)
        L += (-L) % pad_to
        mask = torch.zeros(len(ids), L, dtype=torch.long)
        idx = torch.zeros(len(ids), L, dtype=torch.long)
        for r, i in enumerate(ids):
            idx[r, : len(i)] = torch.tensor(i)
            mask[r, : len(i)] = 1
        return idx, mask

    def encode(self, ids, mask):
        return self.table[ids.cpu()].to(self.device), mask.to(self.device)

    def __call__(self, captions: list[str]):
        return self.encode(*self.tokenize(captions))


def build_text_encoder(kind: str, path: str, device, max_len: int = 128, dim: int = 32):
    if kind == "hash":
        return HashTextEncoder(dim, device, max_len)
    from transformers import AutoTokenizer
    if kind == "ettin":
        return HFTextEncoder(_load_modernbert(path), AutoTokenizer.from_pretrained(path), device, max_len)
    if kind == "t5":
        from transformers import T5EncoderModel
        return HFTextEncoder(T5EncoderModel.from_pretrained(path, torch_dtype=torch.float32),
                             AutoTokenizer.from_pretrained(path), device, max_len)
    raise ValueError(f"unknown text encoder {kind!r} (expected ettin, t5 or hash)")


def compile_forward(module: torch.nn.Module, compiler=None) -> torch.nn.Module:
    """Compile a FROZEN module's forward in place; parameter names and state_dict keys stay unchanged."""
    compiler = compiler or torch.compile
    module.forward = compiler(module.forward)
    return module
