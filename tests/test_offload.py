"""ActOffload (perf_kit/offload.py) is exact: LoRA gradients and losses are bit-identical with and without moving
every activation to pinned host memory and prefetching it back in backward. Needs CUDA (the streams are the point)."""
import pytest
import torch

from perf_kit.models.qwen3_5 import FastQwen35
from perf_kit.offload import ActOffload
from perf_kit.packing import pack
from perf_kit.weights import DictSource

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
eq = pytest.importorskip("test_qwen3_5_equiv")


def _fast(cfg, hf, pm):
    sd = {k.replace("base_layer.", ""): v.detach() for k, v in hf.state_dict().items() if "lora_" not in k}
    fm = FastQwen35(cfg, DictSource(sd), prefix="model.", head_name="lm_head.weight", rank=4, alpha=8,
                    storage="bf16", device="cuda", dtype=torch.float32, master_dtype=torch.float32, impl="ref")
    lora = {k.replace(".default", ""): v.detach() for k, v in pm.state_dict().items() if "lora_" in k}
    fm.lora.load_peft(lora, prefix="base_model.model.")
    return fm


def _run(fm, packed, off):
    fm.offload = off
    fm.lora.zero_grad()
    ce = fm.token_ce(packed)
    ce.sum().backward()
    moved = off.end_step() if off is not None else 0
    return ce.detach().clone(), fm.lora.grad.detach().clone(), moved


@pytest.mark.parametrize("keep_last", [0, 1])
def test_offload_is_bit_identical(keep_last):
    torch.manual_seed(0)
    cfg, hf, pm = eq.build()
    fm = _fast(cfg, hf, pm)
    packed = pack(eq.rows_data()).to("cuda")
    ce0, g0, _ = _run(fm, packed, None)
    off = ActOffload(len(fm.layers), keep_last=keep_last, min_bytes=0)
    for _ in range(2):                                   # second pass reuses pooled pinned buffers
        ce1, g1, moved = _run(fm, packed, off)
        assert moved > 0
        assert torch.equal(ce0, ce1)
        assert torch.equal(g0, g1)
    assert off.pool and not off.handles
