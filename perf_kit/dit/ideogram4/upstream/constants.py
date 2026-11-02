# Copyright Ideogram, Inc. Licensed under the Apache License, Version 2.0 (see ../LICENSE-IDEOGRAM4-CODE).
# From github.com/ideogram-oss/ideogram4, commit 990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2, src/ideogram4/constants.py.
# MODIFIED by LogoLabs.org (vendored as perf_kit's CPU reference): this header only.
SEQUENCE_PADDING_INDICATOR = -1

OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3

# Image grid coordinates start at this offset so they never collide with text token indices
# (text positions start at 0 and never exceed max_text_tokens, which is well below this).
IMAGE_POSITION_OFFSET = 65536

# Layers of Qwen3-VL whose hidden states are concatenated and fed to the transformer.
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
