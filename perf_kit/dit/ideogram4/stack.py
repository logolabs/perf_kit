"""Loading Ideogram 4 for the engine, through Ideogram's own `ideogram4` package.

The weights are NOT part of this repository. They are distributed by Ideogram on Hugging Face (e.g.
ideogram-ai/ideogram-4-fp8, a gated repository) under the Ideogram Non-Commercial Model Agreement; you must accept
that agreement to download them, and a LoRA trained on them is a "Model Derivative" under it. The inference code
(`pip install git+https://github.com/ideogram-oss/ideogram4`) is Apache-2.0.

Ideogram4Stack loads only what training needs:
  * the conditional transformer, its five LoRA-target Linears per block kept as FP8 codes (the engine takes them
    over); every other FP8 Linear is dequantised once (float32 product, rounded to bf16);
  * float32 rotary tables (restore_rope_precision): the package builds the rotary inv_freq in float32 as a
    non-persistent buffer and then casts the whole model to bf16, which rounds it; image tokens sit at position
    65,536, where that rounding becomes phase errors of tens of radians;
  * the Qwen3-VL text encoder and tokenizer, encoded over the caption tokens only (encode_text_prefix: exact, the
    image slots are masked keys and zeroed outputs in the package's encoder);
  * the autoencoder's encoder (encode_images: mean latent, patchified, normalised like the package's decoder
    inverts).
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .engine import LLM_TOKEN_INDICATOR, TARGETS


def restore_rope_precision(model: nn.Module) -> int:
    """Recompute every rotary `inv_freq` in float32 after a low-precision cast. Returns the number repaired."""
    n = 0
    for m in model.modules():
        inv = getattr(m, "inv_freq", None)
        if inv is None or not torch.is_tensor(inv) or not hasattr(m, "head_dim"):
            continue
        base = float(getattr(getattr(model, "config", None), "rope_theta", 5_000_000))
        fixed = 1.0 / (base ** (torch.arange(0, m.head_dim, 2, dtype=torch.float32, device=inv.device) / m.head_dim))
        m.register_buffer("inv_freq", fixed, persistent=False)
        n += 1
    return n


def dequantize_fp8_linear(m: nn.Module, dtype=torch.bfloat16) -> nn.Linear:
    lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None)
    with torch.no_grad():
        lin.weight.copy_((m.weight.to(torch.float32) * m.weight_scale.to(torch.float32).unsqueeze(1)).to(dtype))
        if m.bias is not None:
            lin.bias.copy_(m.bias.to(dtype))
    return lin.to(device=m.weight.device, dtype=dtype)


def dequantize_except_targets(tr: nn.Module, dtype=torch.bfloat16) -> int:
    keep = {id(blk.get_submodule(n)) for blk in tr.layers for n in TARGETS}
    n = 0
    for _, parent in list(tr.named_modules()):
        for cname, child in list(parent.named_children()):
            if hasattr(child, "weight_scale") and id(child) not in keep:
                setattr(parent, cname, dequantize_fp8_linear(child, dtype))
                n += 1
    return n


def _load_tokenizer(repo: str, sub: str):
    """The tokenizer straight from its directory (works offline: AutoTokenizer.from_pretrained(repo, subfolder=)
    first asks for <subfolder>/config.json, which the Ideogram repository does not ship)."""
    import transformers
    from huggingface_hub import hf_hub_download
    d = repo if os.path.isdir(repo) else None
    d = os.path.join(d, sub) if d else os.path.dirname(hf_hub_download(repo, f"{sub}/tokenizer.json"))
    with open(os.path.join(d, "tokenizer_config.json"), encoding="utf-8") as f:
        cls_name = json.load(f).get("tokenizer_class") or "AutoTokenizer"
    return (getattr(transformers, cls_name, None) or transformers.AutoTokenizer).from_pretrained(d)


def _load_text_encoder(repo: str, sub: str, device, dtype):
    """Qwen3-VL through the package's own loaders: its FP8 layout when the config says so, else from_pretrained."""
    from huggingface_hub import hf_hub_download
    from ideogram4.pipeline_ideogram4 import _load_fp8_text_encoder
    from ideogram4.quantized_loading import FP8_TEXT_ENCODER_CONFIG_FLAG
    from transformers import AutoModel
    cfg_file = (os.path.join(repo, sub, "config.json") if os.path.isdir(repo)
                else hf_hub_download(repo, f"{sub}/config.json"))
    with open(cfg_file, encoding="utf-8") as f:
        fp8 = json.load(f).get(FP8_TEXT_ENCODER_CONFIG_FLAG, False)
    if fp8:
        return _load_fp8_text_encoder(repo, device, dtype, text_encoder_subfolder=sub)
    return AutoModel.from_pretrained(repo, subfolder=sub, torch_dtype=dtype, trust_remote_code=True).to(device).eval()


class Ideogram4Stack:
    patch = 2

    def __init__(self, weights_repo: str = "ideogram-ai/ideogram-4-fp8", device="cuda", dtype=torch.bfloat16,
                 max_text_tokens: int = 2048, text: bool = True, vae: bool = True):
        try:
            from huggingface_hub import hf_hub_download
            from ideogram4.modeling_ideogram4 import Ideogram4Config
            from ideogram4.pipeline_ideogram4 import (Ideogram4Pipeline, Ideogram4PipelineConfig, _build_transformer,
                                                      _load_autoencoder, _load_indexed_or_single_state_dict)
        except ImportError as e:
            raise SystemExit("Ideogram 4 needs Ideogram's inference package: "
                             "pip install git+https://github.com/ideogram-oss/ideogram4") from e
        self.device, self.dtype = torch.device(device), dtype
        cfg = Ideogram4PipelineConfig(weights_repo=weights_repo, max_text_tokens=max_text_tokens)
        sd = _load_indexed_or_single_state_dict(cfg.weights_repo, cfg.conditional_index_filename)
        tr = _build_transformer(Ideogram4Config(), sd, self.device, dtype)
        del sd
        self.n_dequantised = dequantize_except_targets(tr, dtype)
        self.n_rope_fixed = restore_rope_precision(tr)
        tok = te = ae = None
        if text:
            tok = _load_tokenizer(cfg.weights_repo, cfg.tokenizer_subfolder)
            te = _load_text_encoder(cfg.weights_repo, cfg.text_encoder_subfolder, self.device, dtype)
        if vae:
            ae = _load_autoencoder(hf_hub_download(repo_id=cfg.weights_repo, filename=cfg.autoencoder_filename),
                                   self.device, dtype)
        for m in (tr, te, ae):
            if m is not None:
                m.requires_grad_(False)
        self.pipe = Ideogram4Pipeline(tr, None, te, tok, ae, cfg, self.device, dtype)
        self.transformer, self.ae = tr, ae
        self.shift, self.scale = self.pipe.latent_shift, self.pipe.latent_scale

    # ------------------------------------------------------------------ text
    def tokenize(self, prompts: list[str]) -> list:
        """[(token ids 1-D LongTensor, n)], exactly as the package's pipeline tokenises (chat template)."""
        return [self.pipe._tokenize(p) for p in prompts]

    @torch.no_grad()
    def encode_text_prefix(self, tokenized: list) -> torch.Tensor:
        """(B, mt, D_llm) float32 features over the left-padded text prefix (mt = longest caption)."""
        B = len(tokenized)
        mt = max(n for _, n in tokenized)
        ids = torch.zeros(B, mt, dtype=torch.long)
        tpos = torch.zeros(B, mt, 3, dtype=torch.long)
        ind = torch.zeros(B, mt, dtype=torch.long)
        for b, (toks, n) in enumerate(tokenized):
            o = mt - n
            ids[b, o:] = toks
            tp = torch.arange(n)
            tpos[b, o:] = torch.stack([tp, tp, tp], 1)
            ind[b, o:] = LLM_TOKEN_INDICATOR
        dev = self.device
        nb = dev.type == "cuda"
        ids, tpos, ind = (x.pin_memory().to(dev, non_blocking=True) if nb else x.to(dev) for x in (ids, tpos, ind))
        return self.pipe._encode_text(ids, tpos, ind)

    # ------------------------------------------------------------------ images
    @torch.no_grad()
    def encode_images(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1] -> (B, H/16 * W/16, 128) normalised latents (the mean of the encoder)."""
        h = self.ae.encoder(pixels.to(self.device, self.dtype))
        z = h.float().chunk(2, dim=1)[0]
        return (patchify(z, self.patch) - self.shift.float()) / self.scale.float()


def patchify(x: torch.Tensor, patch: int = 2) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/p)(W/p), C p p): the exact inverse of the package's _decode un-patchify."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // patch, patch, w // patch, patch).permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, (h // patch) * (w // patch), patch * patch * c)


def tiny_transformer(layers: int = 2, emb: int = 64, heads: int = 2, llm_dim: int = 64, seed: int = 0,
                     dtype=torch.float32, device="cpu") -> nn.Module:
    """A toy Ideogram4Transformer from the vendored upstream definition (random weights, non-trivial norms): what
    the CPU tests and --tiny runs train. Same module tree as the real model."""
    from .upstream.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
    hd = emb // heads
    inv = hd // 2
    s = inv // 3
    cfg = Ideogram4Config(emb_dim=emb, num_layers=layers, num_heads=heads, intermediate_size=2 * emb, adanln_dim=16,
                          in_channels=128, llm_features_dim=llm_dim, mrope_section=(inv - 2 * s, s, s))
    torch.manual_seed(seed)
    tr = Ideogram4Transformer(cfg)
    with torch.no_grad():
        for p in tr.parameters():
            p.add_(0.05 * torch.randn_like(p))
    tr = tr.to(device=device, dtype=dtype)
    restore_rope_precision(tr)
    return tr.requires_grad_(False)
