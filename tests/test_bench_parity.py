"""The benchmark's transformers + PEFT baseline (bench/e2e/hf_peft_train.py) trains the same math as perf_kit.train:
on a tiny Qwen3.5 checkpoint saved to disk (CPU, float32), both harnesses start from the shared init adapter
(bench/e2e/make_init_adapter.py), take the same sequential step sets and produce the same per-step losses and val
loss. This is what makes the speed comparison a comparison of execution engines only."""
import json
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
peft = pytest.importorskip("peft")
tokenizers = pytest.importorskip("tokenizers")

E2E = Path(__file__).resolve().parents[1] / "bench" / "e2e"
sys.path.insert(0, str(E2E))

CHAT = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def _tokenizer(path: Path):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    chars = sorted(set("abcdefghijklmnopqrstuvwxyz0123456789 .,?!\n_<|>"))
    vocab = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "[UNK]": 3}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    from transformers import PreTrainedTokenizerFast
    t = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                unk_token="[UNK]", additional_special_tokens=["<|im_start|>", "<|im_end|>"])
    t.chat_template = CHAT
    t.save_pretrained(str(path))
    return len(vocab)


def _checkpoint(tmp_path: Path) -> Path:
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    d = tmp_path / "model"
    vocab = _tokenizer(d)
    cfg = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
                            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                            linear_conv_kernel_dim=4, rms_norm_eps=1e-6, tie_word_embeddings=False,
                            layer_types=["linear_attention"] * 3 + ["full_attention"],
                            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                             "partial_rotary_factor": 0.25})
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n or "dt_bias" in n or "A_log" in n:
                p.add_(torch.randn_like(p) * 0.1)
    m.save_pretrained(str(d), safe_serialization=True)
    return d


def _data(tmp_path: Path):
    rows = []
    for i in range(24):
        q = f"question {i} " * (1 + i % 4)
        rows.append({"messages": [{"role": "user", "content": q.strip() + "?"},
                                  {"role": "assistant", "content": f"answer {i}, " * (2 + i % 3) + "done."}]})
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:6]))


def _per_step(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "step_losses" in r:
            for j, v in enumerate(r["step_losses"]):
                out[r["step"] - len(r["step_losses"]) + 1 + j] = v
    return out


def _vals(path: Path) -> dict:
    return {r["step"]: r["val_loss"] for r in map(json.loads, path.read_text().splitlines()) if r.get("event") == "val"}


def test_hf_peft_baseline_matches_perf_kit(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    d = _checkpoint(tmp_path)
    _data(tmp_path)
    import make_init_adapter
    import hf_peft_train
    from perf_kit import train as T
    # B != 0 would make the init trivially equal; perturb it so step 1 exercises every LoRA path
    assert make_init_adapter.main(["--model", str(d), "--out", str(tmp_path / "init"), "--rank", "4",
                                   "--alpha", "8"]) == 0
    from safetensors.torch import load_file, save_file
    t = load_file(str(tmp_path / "init" / "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(1)
    t = {k: (v + 0.05 * torch.randn(v.shape, generator=g)) if "lora_B" in k else v for k, v in t.items()}
    save_file(t, str(tmp_path / "init" / "adapter_model.safetensors"))
    common = ["--model", str(d), "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"),
              "--init_adapter", str(tmp_path / "init"), "--rank", "4", "--alpha", "8", "--lr", "5e-3",
              "--warmup", "1", "--max_steps", "3", "--global_batch", "8", "--log_every", "1", "--device", "cpu"]
    assert T.main(common + ["--out", str(tmp_path / "pk"), "--step_order", "sequential", "--no_save",
                            "--impl", "ref", "--storage", "bf16", "--micro_tokens", "200"]) == 0
    assert hf_peft_train.main(common + ["--out", str(tmp_path / "hf"), "--attn", "eager", "--micro_bs", "3"]) == 0
    pk, hf = _per_step(tmp_path / "pk" / "log.jsonl"), _per_step(tmp_path / "hf" / "log.jsonl")
    assert sorted(pk) == sorted(hf) == [1, 2, 3]
    for s in pk:
        assert abs(pk[s] - hf[s]) <= 2e-5 * abs(hf[s]), (s, pk[s], hf[s])
    assert pk[3] < pk[1]                                    # it trains
    vp, vh = _vals(tmp_path / "pk" / "log.jsonl"), _vals(tmp_path / "hf" / "log.jsonl")
    assert sorted(vp) == sorted(vh) == [0, 3]
    for s in vp:
        assert abs(vp[s] - vh[s]) <= 2e-5 * abs(vh[s]), (s, vp[s], vh[s])
