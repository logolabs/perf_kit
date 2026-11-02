import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("WANDB_MODE", "disabled")


import inspect  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def hf_torch_kernels(monkeypatch):
    """transformers routes Qwen3.5's Gated DeltaNet / short-conv functions to fla / causal-conv1d whenever those
    packages import -- even for CPU tensors, where their Triton kernels cannot run. The exactness tests compare
    against transformers' own PyTorch math, so undo the routing (keeping the wrapper's kwarg filtering)."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except Exception:                                   # noqa: BLE001  (transformers < 5)
        return
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        f = getattr(mq, name, None)
        raw = inspect.unwrap(f) if f is not None else None
        if raw is None or raw is f:
            continue
        params = set(inspect.signature(raw).parameters)

        def g(*a, _raw=raw, _p=params, **k):
            return _raw(*a, **{x: y for x, y in k.items() if x in _p})
        monkeypatch.setattr(mq, name, g)
