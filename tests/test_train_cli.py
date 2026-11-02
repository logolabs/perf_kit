"""perf_kit.train end to end on CPU with a tiny Qwen3 (dense) built from transformers: losses finite and falling, PEFT
adapter written, and a run stopped at step 2 and resumed (adapter + optimizer) reproduces the uninterrupted run."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from perf_kit import train as T  # noqa: E402


class Tok:
    name_or_path, chat_template, eos_token = "fake", "fake", "<|im_end|>"

    def __len__(self):
        return 300

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        if text == "<|im_end|>":
            return {"input_ids": [299]}
        return {"input_ids": [ord(c) % 290 for c in text]}

    def convert_tokens_to_ids(self, t):
        return 299 if t == "<|im_end|>" else None


def _setup(tmp_path, monkeypatch):
    cfg = transformers.Qwen3Config(vocab_size=300, hidden_size=48, intermediate_size=80, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=12, tie_word_embeddings=False,
                                   max_position_embeddings=1024)
    torch.manual_seed(0)
    sd = {k: v.detach().clone() for k, v in transformers.Qwen3ForCausalLM(cfg).state_dict().items()}

    def build(args, dev, dtype):
        from perf_kit.models.dense import FastDense
        from perf_kit.weights import DictSource
        return FastDense(cfg, DictSource(sd), prefix="model.", rank=args.rank, alpha=args.alpha, storage="bf16",
                         device=dev, dtype=dtype, impl="ref")
    monkeypatch.setattr(T, "build_model", build)
    monkeypatch.setattr(T, "load_tokenizer", lambda src: Tok())
    rows = [{"messages": [{"role": "user", "content": f"question {i} " * (1 + i % 5)},
                          {"role": "assistant", "content": f"answer number {i} in a sentence"}]} for i in range(40)]
    (tmp_path / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "val.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:6]))


def _args(tmp_path, out, *extra):
    return ["--model", "x", "--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl"), "--out",
            str(tmp_path / out), "--device", "cpu", "--rank", "4", "--alpha", "8", "--lr", "3e-3", "--warmup", "1",
            "--max_steps", "4", "--global_batch", "8", "--ref_micro_bs", "2", "--log_every", "1", "--eval_every", "2",
            "--micro_tokens", "300", *extra]


def _log(p):
    return [json.loads(l) for l in p.read_text().splitlines()]


def test_train_cli_runs_and_resumes_exactly(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert T.main(_args(tmp_path, "full")) == 0
    full = _log(tmp_path / "full" / "log.jsonl")
    losses = {r["step"]: r["loss"] for r in full if "loss" in r}
    assert len(losses) == 4 and all(v == v for v in losses.values()) and losses[4] < losses[1]
    assert (tmp_path / "full" / "final" / "adapter_model.safetensors").exists()
    assert T.main(_args(tmp_path, "a", "--stop_at", "2", "--optim_dir", str(tmp_path / "opt"))) == 0
    assert (tmp_path / "a" / "step-000002").exists()
    assert T.main(_args(tmp_path, "b", "--init_adapter", str(tmp_path / "a" / "step-000002"), "--start_step", "2",
                        "--resume_optim", "--optim_dir", str(tmp_path / "opt"))) == 0
    b = {r["step"]: r["loss"] for r in _log(tmp_path / "b" / "log.jsonl") if "loss" in r}
    assert set(b) == {3, 4} and all(abs(b[k] - losses[k]) < 1e-5 for k in b)
