"""FastDense (reference kernels, float64, CPU) against transformers' Llama / Qwen2 (q/k/v bias) / Qwen3 (q/k norm)
+ PEFT LoRA: per-target CE and every LoRA gradient; packed rows equal rows run alone."""
import pytest
import torch

peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from perf_kit.models.dense import FastDense, DEFAULT_TARGETS   # noqa: E402
from perf_kit.packing import IGNORE, pack                      # noqa: E402
from perf_kit.weights import DictSource                        # noqa: E402


def make(kind):
    common = dict(vocab_size=97, hidden_size=48, intermediate_size=80, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, rms_norm_eps=1e-6, tie_word_embeddings=False, max_position_embeddings=512)
    if kind == "llama":
        cfg = transformers.LlamaConfig(**common, head_dim=12, rope_theta=10000.0,
                                       rope_scaling={"rope_type": "llama3", "factor": 4.0, "low_freq_factor": 1.0,
                                                     "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
        cls = transformers.LlamaForCausalLM
    elif kind == "qwen2":
        cfg = transformers.Qwen2Config(**common, rope_theta=1000.0)
        cls = transformers.Qwen2ForCausalLM
    else:
        cfg = transformers.Qwen3Config(**common, head_dim=16, rope_theta=1000.0)
        cls = transformers.Qwen3ForCausalLM
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    hf = cls(cfg).double()
    with torch.no_grad():
        for n, p in hf.named_parameters():
            if "norm" in n or n.endswith("bias"):
                p.add_(torch.randn_like(p) * 0.1)
    pm = peft.get_peft_model(hf, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                                 target_modules=list(DEFAULT_TARGETS)))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.05)
    return cfg, hf, pm


def rows(seed=0, lens=(30, 61, 4, 90)):
    g = torch.Generator().manual_seed(seed)
    out = []
    for L in lens:
        ids = torch.randint(0, 97, (L,), generator=g)
        lab = ids.clone()
        lab[: max(1, L // 3)] = IGNORE
        out.append((ids.tolist(), lab.tolist(), [2] * L))
    return out


def hf_ce(pm, rs):
    n = max(len(r[0]) for r in rs)
    ids = torch.zeros(len(rs), n, dtype=torch.long)
    lab = torch.full((len(rs), n), IGNORE, dtype=torch.long)
    att = torch.zeros(len(rs), n, dtype=torch.long)
    for i, (a, b, _) in enumerate(rs):
        ids[i, :len(a)], lab[i, :len(b)], att[i, :len(a)] = torch.tensor(a), torch.tensor(b), 1
    lg = pm(input_ids=ids, attention_mask=att).logits.double()[:, :-1]
    tgt = lab[:, 1:]
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE,
                                           reduction="none").view(tgt.shape)
    return torch.cat([ce[i, :len(r[0]) - 1][tgt[i, :len(r[0]) - 1] != IGNORE] for i, r in enumerate(rs)])


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
def test_dense_matches_hf_and_peft(kind):
    cfg, hf, pm = make(kind)
    rs = rows()
    ref = hf_ce(pm, rs)
    ref.sum().backward()
    sd = {k.replace("base_layer.", ""): v.detach() for k, v in hf.state_dict().items() if "lora_" not in k}
    fm = FastDense(cfg, DictSource(sd), prefix="model.", rank=4, alpha=8, storage="bf16", device="cpu",
                   dtype=torch.float64, master_dtype=torch.float64, impl="ref")
    lora = {k.replace(".default", ""): v.detach() for k, v in pm.state_dict().items() if "lora_" in k}
    fm.lora.load_peft(lora)
    ce = fm.token_ce(pack(rs))
    ce.sum().backward()
    assert torch.allclose(ce.detach(), ref.detach(), rtol=1e-6, atol=1e-6), (ce - ref).abs().max()
    grads = {n.replace(".default", ""): p.grad for n, p in pm.named_parameters() if "lora_" in n}
    for name, m in fm.lora.slots.items():
        for mine, ab in ((m.a_grad, "A"), (m.b_grad, "B")):
            theirs = grads[f"base_model.model.{name}.lora_{ab}.weight"]
            assert (mine - theirs).abs().max() <= 1e-5 * (theirs.abs().max() + 1e-12), (name, ab)
    apart = torch.cat([fm.token_ce(pack([r])) for r in rs])
    assert torch.allclose(ce.detach(), apart.detach(), rtol=1e-10, atol=1e-10)
