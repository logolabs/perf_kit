"""The fast Qwen3.5 forward/backward (reference kernels, float64, CPU) against transformers' own model + PEFT LoRA:
per-target CE and every LoRA gradient must match. Packed rows must equal the padded batch row by row."""
import pytest
import torch

mq = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
peft = pytest.importorskip("peft")

from perf_kit.models.qwen3_5 import FastQwen35, DEFAULT_TARGETS   # noqa: E402
from perf_kit.packing import IGNORE, pack                          # noqa: E402
from perf_kit.weights import DictSource                            # noqa: E402


def tiny_cfg():
    from transformers import Qwen3_5TextConfig
    return Qwen3_5TextConfig(vocab_size=101, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                             linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                             linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=False,
                             layer_types=["linear_attention"] * 3 + ["full_attention"],
                             rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                              "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
                                              "mrope_interleaved": True})


def rows_data(seed=0, lens=(37, 70, 5, 129)):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for L in lens:
        ids = torch.randint(0, 101, (L,), generator=g)
        p = max(1, L // 3)
        lab = ids.clone()
        lab[:p] = IGNORE
        seg = torch.zeros(L, dtype=torch.long)
        seg[p:] = 2
        rows.append((ids.tolist(), lab.tolist(), seg.tolist()))
    return rows


def build(seed=0):
    torch.manual_seed(seed)
    cfg = tiny_cfg()
    cfg._attn_implementation = "eager"            # CPU float64 SDPA is broken for causal rows > 32 (torch 2.5)
    hf = mq.Qwen3_5ForCausalLM(cfg).double()
    with torch.no_grad():                      # non-trivial norms / gates (init is ones / zeros)
        for n, p in hf.named_parameters():
            if "norm" in n or "dt_bias" in n or "A_log" in n:
                p.add_(torch.randn_like(p) * 0.1)
    lc = peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=list(DEFAULT_TARGETS), bias="none")
    pm = peft.get_peft_model(hf, lc)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.copy_(torch.randn_like(p) * 0.05)
    return cfg, hf, pm


def hf_token_ce(pm, rows):
    n = max(len(r[0]) for r in rows)
    ids = torch.zeros(len(rows), n, dtype=torch.long)
    lab = torch.full((len(rows), n), IGNORE, dtype=torch.long)
    att = torch.zeros(len(rows), n, dtype=torch.long)
    for i, (a, b, _) in enumerate(rows):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    logits = pm(input_ids=ids, attention_mask=att).logits.double()
    tgt = lab[:, 1:]
    lg = logits[:, :-1]
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE,
                                           reduction="none").view(tgt.shape)
    out = []
    for i, r in enumerate(rows):
        L = len(r[0])
        m = tgt[i, :L - 1] != IGNORE
        out.append(ce[i, :L - 1][m])
    return torch.cat(out)


def fast_from(cfg, hf, pm, storage="bf16"):
    sd = {k: v.detach() for k, v in hf.state_dict().items() if "lora_" not in k}
    sd = {k.replace("base_layer.", ""): v for k, v in sd.items()}
    fm = FastQwen35(cfg, DictSource(sd), prefix="model.", head_name="lm_head.weight", rank=4, alpha=8,
                    storage=storage, device="cpu", dtype=torch.float64, master_dtype=torch.float64, impl="ref")
    lora = {k.replace(".default", ""): v.detach() for k, v in pm.state_dict().items() if "lora_" in k}
    n = fm.lora.load_peft(lora, prefix="base_model.model.")
    assert n == 2 * len(fm.lora.slots)
    return fm


def test_forward_ce_matches_hf():
    cfg, hf, pm = build()
    rows = rows_data()
    ref = hf_token_ce(pm, rows)
    fm = fast_from(cfg, hf, pm)
    ce = fm.token_ce(pack(rows))
    assert ce.shape == ref.shape
    assert torch.allclose(ce, ref, rtol=1e-6, atol=1e-6), (ce - ref).abs().max()


@pytest.mark.parametrize("ckpt", [None, [True, False, True, True]])
def test_lora_grads_match_peft(ckpt):
    cfg, hf, pm = build(1)
    rows = rows_data(1)
    ref = hf_token_ce(pm, rows)
    ref.sum().backward()
    fm = fast_from(cfg, hf, pm)
    fm.lora.zero_grad()
    ce = fm.token_ce(pack(rows), ckpt=ckpt)
    ce.sum().backward()
    assert torch.allclose(ce.detach(), ref.detach(), rtol=1e-6, atol=1e-6)
    grads = {n.replace(".default", ""): p.grad for n, p in pm.named_parameters() if "lora_" in n}
    worst = 0.0
    for name, m in fm.lora.slots.items():
        ga = grads[f"base_model.model.{name}.lora_A.weight"]
        gb = grads[f"base_model.model.{name}.lora_B.weight"]
        for mine, theirs in ((m.a_grad, ga), (m.b_grad, gb)):
            scale = theirs.abs().max().item() + 1e-12
            worst = max(worst, (mine - theirs).abs().max().item() / scale)
    assert worst < 1e-5, worst


def test_packed_equals_rows_separately():
    cfg, hf, pm = build(2)
    rows = rows_data(2)
    fm = fast_from(cfg, hf, pm)
    together = fm.token_ce(pack(rows))
    apart = torch.cat([fm.token_ce(pack([r])) for r in rows])
    assert torch.allclose(together, apart, rtol=1e-10, atol=1e-10)
