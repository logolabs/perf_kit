# Licences in perf_kit/dit/ideogram4

Three different things are involved here, under three different terms.

1. **Ideogram's code (`upstream/`).** `modeling_ideogram4.py`, `constants.py` and `scheduler.py` are copied from
   Ideogram 4's inference code, [github.com/ideogram-oss/ideogram4](https://github.com/ideogram-oss/ideogram4) at
   commit `990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2`, Copyright Ideogram, Inc., licensed under the Apache License 2.0
   ([`LICENSE-IDEOGRAM4-CODE`](LICENSE-IDEOGRAM4-CODE), a verbatim copy of that repository's `LICENSE.md`). The
   header of each file says what was changed (a comment header and, in `modeling_ideogram4.py`, one import line).
   They are the CPU reference the engine is tested against and the model class of tiny / synthetic runs.

2. **perf_kit's engine (`engine.py`, `ops.py`, `kernels.py`, `stack.py`, `data.py`, `train.py`).** LogoLabs code,
   under this repository's Apache-2.0 licence. `engine.py` and `ops.py` re-implement the computation of the
   transformer in `upstream/modeling_ideogram4.py` operation by operation (packed, with a hand-written backward); as
   a derivative of Ideogram's Apache-2.0 code they keep the attribution above.

3. **Ideogram 4's weights: NOT included, NOT covered by any licence in this repository.** The weights (for example
   the gated Hugging Face repository `ideogram-ai/ideogram-4-fp8`) are licensed by Ideogram, Inc. under the
   **Ideogram Non-Commercial Model Agreement** (`model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL` in Ideogram's
   repository, last updated June 3, 2026). You must accept that agreement to download them. In its terms:
   - a fine-tuned version of the model, such as a LoRA trained with this engine, is a **Model Derivative**;
   - the model and Model Derivatives may be used, reproduced and distributed **for Non-Commercial Purposes only**;
     for a for-profit entity that means testing, evaluation, or research and development in a non-production
     environment, and any training or fine-tuning for commercial use is excluded;
   - distributing a Model Derivative requires passing on a copy of the agreement, the attribution notice it
     specifies, terms at least as restrictive, prominent notices on modified files, and not implying that Ideogram
     made or endorsed it.

   Commercial use needs a separate agreement with Ideogram. Read the agreement itself; this summary is not legal
   advice and does not replace it.
