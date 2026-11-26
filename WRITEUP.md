# Making LoRA on a hybrid Gated-DeltaNet LLM fast on GH200, measured

*perf_kit 0.1.0, November 2026. Every number below is reproducible from this repository (`results/raw/` holds the
logs, `bench/analysis/` turns them into the tables and figures). Where a number comes from earlier internal runs on
private data, it is labelled as such.*

**TL;DR.** On Qwen3.5-9B-Base (LoRA r16 on every projection, public UltraChat data, identical 64-row steps for
every framework), perf_kit trains at **12,523 tokens/s per GH200 on 1 GPU and 12,198 per GPU on one
4-GPU node, 37.6% MFU** (conservative FLOP count, 989 TFLOPS peak; ~52% at the clock the
power-capped GPUs actually ran). The best tuned baseline we found is ms-swift without gradient checkpointing on 1
GPU (7,902 tokens/s, 1.58x slower) and ms-swift 4.5.3 (padding-free, FA2, Liger) on 4 GPUs (7,214 tokens/s/GPU, 1.69x).
Losses match transformers + PEFT step by step (differences < 6e-4 on a loss of ~0.95; end-of-run val equal to
0.03%). Most of the gain comes from removing work around the GEMMs, not from new kernels; several things we
expected to help did not, and are listed with numbers. All of it is GH200-only: no H100 numbers.

The same approach, applied to the two image models LogoLabs trains, is in sections 6 and 7: a LoRA engine for the
Ideogram 4 diffusion transformer (1.86x the reference trainer on one 4-GPU node, 35.9% MFU) and the training loop of
Agate, an FCDM text-to-image model (+24% images per second). Those numbers are internal measurements, on private data
or with logs that are not in this repository, and are labelled as such.

![throughput](figures/throughput.png)

## 1. The problem

Qwen3.5 (and the larger Qwen3.8-27B we trained most) is a *hybrid* decoder: three of every four layers are Gated
DeltaNet (GDN) linear-attention layers (a gated delta-rule recurrence computed in chunks, preceded by a short causal
convolution), the fourth is gated softmax attention. LoRA fine-tuning freezes all base weights and trains low-rank
adapters on every projection. That combination has a cost structure most training stacks are not built for:

- **The useful FLOPs are almost all frozen GEMMs.** Per token the frozen projections cost 2N in the forward and 2N
  for the input gradient; no weight gradient is needed. LoRA adds ~0.8% of the FLOPs at r16 on the 9B, ~5% at r128
  on the 27B. The job is to keep the frozen GEMMs running back to back; everything else on the GPU timeline is
  overhead.
- **The standard stack runs GDN padded.** transformers uses fla's Triton kernels for the GDN recurrence when fla
  is installed, but its short convolution falls back to a PyTorch `conv1d` unless the `causal-conv1d` package is
  present (no aarch64 wheel), and that fallback cannot respect row boundaries in packed batches. So padded batches
  it is.
- **PEFT's LoRA wrappers are launch-heavy.** Each adapted Linear becomes several kernels plus autograd bookkeeping
  in both directions, and fp32 adapters under bf16 autocast are re-cast at every forward (and again in recompute).
- **Grace dispatches slowly.** PyTorch op dispatch on the Grace CPU is reported at ~2.5x the cost of a fast x86
  host (Transformer Engine issue #2053), so launch-bound code that is invisible on an x86 H100 node shows up as GPU
  idle time on GH200.
- **The GPUs run power-capped.** The GH200 module limit on our cluster is 660 W (GPU + Grace + memory). Under
  training load the SM clock sits at 1.4-1.8 GHz, not the 1.98 GHz the 989 TFLOPS bf16 peak assumes, and the
  software power cap is active in 82-100% of samples in every arm. Faster code draws more power and runs at a *lower*
  clock (perf_kit ~1.43 GHz, the padded baselines 1.55-1.80 GHz). We therefore report MFU against the nominal peak
  and a clock-adjusted MFU (MFU x 1980 / measured median SM MHz).

Before writing any code we profiled the plain transformers + PEFT trainer on the 27B: 11-14% MFU, GPU idle 29% of
the time, copies and casts 25% of kernel time, while the GEMMs themselves already ran at ~620 TFLOPS. The kernels
were fine; everything around them was not.

## 2. What perf_kit does

A LoRA step on a frozen base is dominated by the frozen projections' GEMMs (forward + input gradient, 4N FLOPs
per token). Everything perf_kit does either keeps those GEMMs fed or removes work around them. Each item below is
exact (same math as transformers + PEFT up to float summation order) unless marked otherwise, and is covered by a
CPU float64 test against transformers + PEFT.

1. **Fused projections + fused LoRA, no PEFT wrappers.** Projections that read the same input (q/k/v;
   gate/up; the four Gated DeltaNet input projections) run as one GEMM over a concatenated weight. All LoRA A
   matrices of a group run as one GEMM, each B is added with one `addmm` into the base output. LoRA weights are not
   autograd leaves: the backward accumulates their gradients straight into fp32 buffers with
   `torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)` (one cuBLAS call, no bf16 rounding of the
   per-micro-batch gradient, no AccumulateGrad nodes, no per-step re-cast of fp32 LoRA weights under autocast).
2. **One flat LoRA store.** All adapters live in one fp32 master tensor (+ grad, + bf16 compute copy with B
   pre-scaled by alpha/r). The optimizer is a single fused AdamW over one tensor, the gradient all-reduce is one
   NCCL call, refreshing the compute copies after a step is two kernels.
3. **Varlen packing without padding.** A micro-batch is one packed sequence + `cu_seqlens`: flash-attn varlen
   for the attention layers, fla's chunked Gated DeltaNet (native grouped value heads, gate and beta computed
   inside the kernel) and fla's causal conv with `cu_seqlens`. Targets never cross row boundaries and positions
   restart per row. (transformers' PyTorch conv fallback ignores row boundaries in padding-free batches, so a
   packed row would see its neighbour's last 3 tokens; perf_kit's packed result equals the rows run one by one,
   tested.)
4. **Step sets are separate from layout.** The loss is normalised by the step's global target count, so the
   gradient depends only on which rows the step trains, not on how they are split over ranks and micro-batches.
   The step set comes from a reproducible sampler; rows are then dealt to ranks longest-first onto the least
   loaded rank (LPT) and packed first-fit-decreasing into token-budget micro-batches. On our long-tailed 4-GPU
   workload this cut the rank straggler tax from 12-23% of every step to under 1%.
5. **Recompute only what memory requires, measured at start-up.** Per micro-batch, a plan checkpoints just
   enough layers (spread evenly) to fit the activation budget. The per-token activation size is measured on the
   real model with one probe forward when training starts (new in this release; the previous fixed estimate of
   0.25 MiB/token/layer over-stated the Qwen3.5-9B's measured 0.157-0.158 by ~60% and made recompute start with a
   third of the HBM still free).
6. **Fused small ops.** Liger RMSNorm and residual-add + RMSNorm (one kernel each way), an own SwiGLU kernel
   that reads the fused gate|up output in place and writes the fused gradient, fla's gated RMSNorm,
   `torch.split` instead of slicing (a slice's backward zero-fills and copies a full-size gradient), a chunked
   frozen LM head + cross-entropy that never materialises `[tokens x vocab]` logits and runs only at target
   positions.
7. **No host syncs inside a step.** Losses and counters accumulate on the GPU and are read at log steps;
   micro-batches are packed and pinned on a CPU thread and copied on a side stream.
8. **Block-FP8 checkpoints kept as FP8 codes** (`--storage fp8`), dequantised bit-exactly to bf16 right before
   each GEMM by a Triton kernel (~2.9 TB/s on GH200): the same bf16 weights in half the memory. At 27B this freed
   ~27 GB and removed nearly all recompute. (The 9B benchmark checkpoint is bf16, so this does not apply there.)

## 3. What each piece is worth (ablations)

One model load, then the same 16 steps (64 rows each) per configuration from the same LoRA init with a fresh
optimizer; median step time of the last 10 steps; every step CUDA-synchronised; 1 GH200; 8,192-token
micro-batches; two independent jobs on different nodes for most configurations. Reference: the full engine
(2.90 s/step, median of two runs). The first configuration of each job is excluded (it absorbs the process's JIT
warm-up and ran 4-5% slow both times).

![ablation](figures/ablation.png)

| removed / replaced (one at a time) | step time vs full engine | n |
|---|---|---|
| recompute every layer (what a standard trainer does) | +41.2% (+41.2) | 1 |
| per-row PyTorch conv instead of fla's varlen causal conv | +20.2% (+20.3, +20.1) | 2 |
| PyTorch RMSNorm / add + norm instead of Liger | +14.5% (+14.7, +14.4) | 2 |
| per-row eager attention instead of flash-attn varlen | +12.5% (+12.4, +12.6) | 2 |
| PyTorch SwiGLU instead of the fused gate|up kernel | +9.9% (+9.9) | 1 |
| PyTorch gated RMSNorm instead of fla's | +7.5% (+7.5) | 1 |
| bf16 LoRA grads + separate add instead of fp32-accumulating addmm | +0.7% (+0.7) | 1 |
| flash-attention 3 instead of 2 | -0.1% (-0.1, -0.1) | 2 |
| FlashQLA GDN kernels instead of fla | -1.0% (-1.1, -0.8) | 2 |
| GDN gate / beta in PyTorch instead of inside fla's kernel | -4.9% (-5.7, -4.1) | 2 |

Per-step losses agree across all configurations to the 3rd-4th decimal (bf16 summation order), which is the
numerical check that each piece is exact.

Two results we did not expect:

- **fla's in-kernel gate is slower here.** Computing the GDN gate `g = -exp(A_log) * softplus(a + dt_bias)` and
  `beta = sigmoid(b)` in PyTorch and passing them to fla was 4.1% and 5.7% *faster* (two runs) than fla's
  `use_gate_in_kernel` path, which perf_kit used by default. We left the default alone because the 27B was not
  re-measured; `PK_DISABLE=gdn_fused_gate` switches it.
- **FA3 and FlashQLA change nothing end to end on the 9B** (-0.1% and -1.0%, inside run-to-run noise), although FA3
  is 12% faster than FA2 as an isolated kernel at the 27B's shapes and FlashQLA is 17% slower than fla there. At 9B
  attention is ~1.5% and GDN ~8% of GPU time; a kernel win there cannot move the step by much.


Two bugs we found while preparing the release (both fixed, both with tests):

- **The "reference" GDN path was not the reference.** transformers 5.16 routes the name
  `torch_chunk_gated_delta_rule` to fla's Triton kernel whenever fla is importable, even for CPU tensors. perf_kit's
  pure-PyTorch reference imported that name, so on machines with fla installed the "fast vs reference" layer
  benchmark compared fla with fla (an earlier internal 1.8x "fast vs ref" figure was really ~23x: 46 vs 1,086 ms per
  four 27B-width layers) and the CPU exactness tests crashed. The reference now unwraps to transformers' own
  PyTorch function, and the tests do the same for transformers' model.
- **The recompute planner over-estimated activations** (next section).

### The planner fix

perf_kit decides per micro-batch how many layers to recompute so that stored activations fit the free HBM. It used
a fixed 0.25 MiB per token per layer; the 9B actually stores 0.157-0.158 MiB (measured at start-up in every run),
so recompute kicked in at ~7.3k tokens with a third of the HBM free. The release measures it on the real model with
one probe forward (`engine.measure_act_mb`, CUDA allocator count after a warm-up forward, tested against a real
step) and plans with the measurement x 1.05.

![micro_sweep](figures/micro_sweep.png)

At 12,288-token micro-batches the old estimate recomputes 28.5% of the layers (3.03-3.09 s/step); the measured plan
recomputes none (2.70-2.76 s/step): **11-12% faster at the same batch**, and it makes the fastest setting (10-12k
tokens) reachable without hand-tuning. Past ~12.4k tokens the HBM really is full and recompute starts (16k tokens:
19.5% recomputed, 3.14 s). On the 27B the same measurement gives 0.216 MiB/token/layer, i.e. no recompute up to
~3.7k tokens on one GPU with FP8-stored weights.

## 4. What did not work (and why)

| idea | what we measured | verdict |
|---|---|---|
| FP8 forward GEMMs (vLLM CUTLASS block-scaled, `FastQwen35(fp8_forward=True)`, API only) | 1.1-1.5x per GEMM in isolation incl. activation quantisation, ~3.7% relative output error; in situ at 27B the step got slower (9.5 vs 6.9 s) | off by default, lossy, A/B only |
| FP8 via `torch._scaled_mm` | block (1x128 / 128x128) scales rejected; one accepted layout returned garbage (rel. err 1.7e5); row-wise worked but the whole step ran at 0.71x of bf16 | not used |
| torchao float8 row-wise | 3-4x slower than bf16 per projection (eager) | not used |
| FlashQLA (TileLang GDN kernels) | 17% slower than fla's GDN fwd+bwd at the 27B shapes (kernel A/B), slightly larger error; end to end on the 9B -1.0% step time (n=2, inside noise) | `PK_GDN_BACKEND=flashqla`, not default |
| flash-attn 3 | 12% faster attention kernel at the 27B shapes, same error; end to end on the 9B -0.1% (n=2): attention is ~1.5% of GPU time | opt-in `PK_ATTN_BACKEND=fa3` |
| fla's in-kernel GDN gate (`use_gate_in_kernel`, our default) | 4.4-5.3% *slower* than computing gate / beta in PyTorch on the 9B (n=2), identical losses | kept default until re-measured at 27B; `PK_DISABLE=gdn_fused_gate` |
| fp32-accumulating LoRA-gradient addmm | replacing it with bf16 grads + add: +0.7% step time at r16 (noise); kept for accuracy (no bf16 rounding of per-micro-batch grads), not speed | kept |
| CUDA graphs / regional compile | GPU idle ~2-5% with the fused path at 27B | not pursued at 27B (might pay at small models) |
| activation offload to Grace memory (`--act_offload`) | bit-identical (tested); the only in-situ trial (27B, 1 GPU) ran out of host memory (~102 GB per GPU job) and, with 1.5x micro-batches, out of HBM | experimental, no measured speed-up |
| adaptive recompute without freeing memory first | slower than fixed recompute in the old trainer | superseded by the measured plan |

## 5. The comparison

### Setup

- **Model / data**: Qwen3.5-9B-Base (bf16 checkpoint), LoRA r16 / alpha 32 on every projection of the text decoder
  (in_proj_qkv, in_proj_z, out_proj, q/k/v/o, gate/up/down: 40,108,032 trainable parameters, identical in every
  framework, checked). First user/assistant exchange of each conversation of
  [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT), rows in file
  order; 530 tokens per row on average, 64% of tokens trained; Qwen chat template with thinking disabled.
- **Identical work**: step s trains rows `[64 s, 64 s + 64)` in every framework; the same token ids (ms-swift's
  first 256 ids checked equal; swift additionally trains the newline after `<|im_end|>`, +1 token per row, counted
  in its throughput); the same init adapter (checksum logged by every harness); lr 2e-4, 6 warm-up steps, cosine to
  0.1x over 60 steps (swift: transformers' `cosine_with_min_lr`, offset by one step); AdamW (0.9, 0.99), no weight
  decay, clip 1.0; loss normalised by the step's global target count.
- **Each framework at its best**: we swept micro-batch size and gradient checkpointing for each baseline (the
  failed or slower settings are in `results/summary.md`) and gave them the same fast kernels where they accept
  them: flash-attention 2 (sm_90 build), Liger RMSNorm / SwiGLU / fused linear-CE, fla's GDN kernels. Padded arms
  sort each step's rows by length before forming micro-batches. Unsloth uses its own patched model, LoRA wrapper
  and offloaded checkpointing inside the same (padded) training loop.
- **Launch**: every arm the same way: one process per GPU pinned to its own Grace (`srun --cpu-bind=cores`,
  `numactl --localalloc`), energy-monitored with perun, nvidia-smi sampled at 1 Hz.
- **Metric**: real (unpadded) tokens per second per GPU over steps 41-60 of a 60-step run; MFU with
  `(4 N_proj + 6 N_lora) x tokens + 4 N_head x targets` FLOPs (attention-score and GDN-state FLOPs not counted, so all
  MFUs are slight under-estimates) against 989 TFLOPS. Repeats are separate jobs on different nodes.

### Results

| GPUs | trainer | setting | n | tokens/s/GPU steps 41-60 (min-max) | steps 11-60 | MFU | clock-adj. | SM MHz | padding | peak alloc GB | val @60 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | perf_kit (warm) | 12288-token packed micro-batches | 2 | 12,523 (12,523-12,524) | 12,484 | 38.6% | 53.9% | 1418 | 0% | 81.0 | 0.9677 |
| 1 | perf_kit (cold) | 12288-token packed micro-batches | 4 | 9,993 (9,918-10,077) | 8,065 | 30.8% | 42.8% | 1425 | 0% | 81.0 | 0.9677 |
| 1 | ms-swift 4.5.3 (padding-free, FA2, Liger) | 8 rows/micro-batch, no gradient checkpointing | 3 | 7,902 (7,728-8,010) | 6,213 | 24.4% | 31.2% | 1548 | 0% | 65.9 | 0.9678 |
| 1 | Unsloth 2026.10.11 | 8 rows/micro-batch, gradient checkpointing | 3 | 7,115 (7,070-7,158) | 7,073 | 21.9% | 29.7% | 1465 | 14% | 22.7 | 0.9675 |
| 1 | Unsloth 2026.10.11 | 16 rows/micro-batch, gradient checkpointing | 1 | 6,924 (6,924-6,924) | 6,134 | 21.4% | 29.1% | 1455 | 26% | 27.7 | 0.9675 |
| 1 | transformers + PEFT + FA2 + Liger | 4 rows/micro-batch, no gradient checkpointing | 3 | 5,461 (5,197-5,635) | 5,455 | 16.9% | 19.3% | 1738 | 6% | 82.5 | 0.9678 |
| 1 | transformers + PEFT + FA2 + Liger | 16 rows/micro-batch, gradient checkpointing | 3 | 4,823 (4,779-4,868) | 4,766 | 14.9% | 18.9% | 1555 | 26% | 71.4 | 0.9677 |
| 1 | transformers + PEFT (SDPA) | 8 rows/micro-batch, gradient checkpointing | 3 | 4,511 (4,476-4,535) | 4,486 | 13.9% | 17.2% | 1605 | 14% | 73.4 | 0.9676 |
| 4 | perf_kit (warm) | 12288-token packed micro-batches | 2 | 12,198 (12,084-12,313) | 12,161 | 37.6% | 51.7% | 1440 | 0% | 69.3 | 0.9676 |
| 4 | ms-swift 4.5.3 (padding-free, FA2, Liger) | 8 rows/micro-batch, no gradient checkpointing | 2 | 7,214 (7,193-7,235) | 6,437 | 22.2% | 27.8% | 1582 | 0% | 64.4 | 0.9678 |
| 4 | perf_kit (cold) | 12288-token packed micro-batches | 3 | 6,278 (6,233-6,343) | 6,785 | 19.4% | 26.4% | 1455 | 0% | 69.3 | 0.9677 |
| 4 | transformers + PEFT + FA2 + Liger | 4 rows/micro-batch, no gradient checkpointing | 2 | 4,908 (4,882-4,934) | 4,836 | 15.1% | 17.0% | 1762 | 17% | 58.0 | 0.9677 |
| 4 | transformers + PEFT (SDPA) | 8 rows/micro-batch, gradient checkpointing | 2 | 3,190 (3,174-3,207) | 3,057 | 9.8% | 11.5% | 1688 | 34% | 52.2 | 0.9678 |
| 4 | transformers + PEFT + FA2 + Liger | 16 rows/micro-batch, gradient checkpointing | 2 | 2,697 (2,681-2,713) | 2,609 | 8.3% | 9.6% | 1718 | 54% | 71.7 | 0.9678 |

Settings tried and dropped (see results/summary.md): transformers + PEFT with 16 rows (OOM: fp32 logits), FA2 + Liger without checkpointing at 8 / 16 rows (OOM), ms-swift with checkpointing at 8 / 16 rows (4.6-5.1k tokens/s, 24-step tuning runs).

**Cold vs warm.** perf_kit's first process on a node spends its first ~50 steps JIT-compiling Triton kernel
specialisations for the packed lengths it meets (fla, Liger and our kernels); later processes on the node reuse the
on-disk cache and are at full speed from step ~10. Steps 41-60 of a cold run still contain compilation (row
"cold"); a warm-cache run is the steady state of a long job. ms-swift and Unsloth also warm up for 20-30 steps (their
steps 41-60 are past it); the transformers arms are flat from step 10. In a multi-hour run the compile cost is a
few minutes once; for short runs, point `TRITON_CACHE_DIR` at persistent storage.

**Padding.** The padded arms (transformers + PEFT and Unsloth in the same loop) spend 6% (FA2 + Liger, 4 rows), 14%
(SDPA, 8 rows), 14% (Unsloth, 8 rows), 26% (FA2 + Liger, 16 rows) of their token slots on padding on 1 GPU, although
each step's rows are sorted by length before forming micro-batches (the best a padded trainer can do without
changing which rows a step trains). Even with zero padding (measured throughput divided by the real-token share) the
best of them (Unsloth, 8 rows) would reach ~8,225 tokens/s on 1 GPU; perf_kit (warm) is still 1.52x faster.

**Scaling.** perf_kit keeps its per-GPU throughput from 1 to 4 GPUs (LPT dealing + one all-reduce per step).
The padded arms lose 10-45% per GPU at 4 GPUs because each rank then sees only 16 rows of a 64-row step, so its
micro-batches pad more (up to 54% padding) and the fixed per-step costs are amortised over less work.

![loss_parity](figures/loss_parity.png)

**Same math.** Per-step training losses of perf_kit and transformers + PEFT (with and without FA2 + Liger)
differ by less than 6e-4 at every step, and those of perf_kit and Unsloth by at most 6.3e-4 (bf16 noise), start from the same value (step-1 loss
0.9651-0.9656) and end at the same validation loss (0.96749-0.96778 across all five frameworks; base model
1.0454-1.0456). A CPU test runs the same two harnesses on a tiny Qwen3.5 checkpoint in float32 and requires equal
per-step losses to 2e-5 relative (`tests/test_bench_parity.py`); perf_kit's kernels are separately tested against
transformers + PEFT in float64 (CE and every LoRA gradient, 1e-6 / 1e-5).

### 27B on public data

perf_kit only (no baseline re-run at 27B; see `results/prior_27b_private_data.json` for the earlier comparison on
private data): Qwen3.8-27B with its block-FP8 checkpoint kept as FP8 codes (dequantised exactly per GEMM, 28.5 GB
of frozen weights), 1 GH200, same data and LoRA r16: **3,671 tokens/s, 37.6% MFU** (53% clock-adjusted at
1,395 MHz) with 3,600-token micro-batches and no recompute; 6,144-token micro-batches need 40% recompute and are
12% slower.

Earlier (October 2026, private SFT data, r128, 4 GH200, 300 identical steps): transformers + PEFT 5.155 s/step,
ms-swift 3.70 s/step (with gradient checkpointing), perf_kit 1.82 s/step (36.8% MFU by the 4 x 27e9 convention),
validation loss 0.6866 / 0.6878 / 0.6863. The swift number there used gradient checkpointing; on the 9B, swift is
1.7x faster without it, which we did not try at 27B, so read that comparison as an upper bound on perf_kit's
advantage at 27B.

### Where the time goes now

A torch.profiler capture of two steps (1 GH200, 9B, 8,192-token micro-batches): GEMMs 69% of GPU kernel time, fla
GDN 8%, copies/casts 7%, causal conv 3.4%, norms 3.2%, everything else under 3% each; ~20,200 kernel launches per
step, and the host spends about as long issuing ops as the GPU spends executing them. At 9B on one GH200 perf_kit is
**launch-bound on the Grace CPU**: the remaining lever is CUDA graphs / fewer launches, not faster kernels. (At 27B
the larger GEMMs hide the host; GPU idle was ~2-5% there.) The GEMMs run at ~480 TFLOPS over the step, about 68% of
what the power-capped clock allows.

## 6. Ideogram 4 DiT LoRA (`perf_kit/dit/ideogram4`)

*Internal measurements (October 2026) on a private image + caption dataset, against LogoLabs' reference trainer;
not reproducible from this repository. Numbers: `results/internal_ideogram4_2026-10.json`, research notes
2026-10-31.*

![ideogram4](figures/ideogram4.png)

Ideogram 4 is a 9.3B-parameter flow-matching diffusion transformer: 34 blocks of adaLN-modulated attention + SwiGLU
MLP over one packed sequence of Qwen3-VL text features and 2x2-patchified VAE latents, distributed as a weight-only
FP8 checkpoint. LoRA on attention.qkv / o and feed_forward.w1-3 again leaves almost all useful FLOPs in frozen GEMMs.
LogoLabs' reference trainer already had its own fixes (flash-attention 3 varlen, caption-only text encoding, a tuned
activation-checkpoint budget: 1.34x on one GPU and 1.37x on four, MFU 18.4% to 24.7%; jobs 3650401 / 3650402 /
3651129, 2026-10-10) and still sat at ~20% MFU at 1024 x 1024.

**What the engine does.** The perf_kit recipe, transposed to a DiT:

1. **Packed real tokens.** The upstream layout left-pads text to the longest caption; the engine drops the padding and
   runs every GEMM and elementwise op over real tokens, with flash-attention 3 varlen (bidirectional per sample).
2. **Fused projections + fused (binned) LoRA, no autograd.** One GEMM per group (qkv, o, w1|w3, w2); all adapters of
   a group in one GEMM, including several timestep-binned adapters blended per sample; LoRA gradients accumulated in
   fp32 by `addmm(..., out_dtype=float32, out=acc)`.
3. **A hand-written backward with lean activations.** Per block it keeps the float32 residual input, qkv, the
   attention output + log-sum-exp, and the o / w1|w3 / w2 outputs; norms, modulation, rotary and SwiGLU are recomputed
   in the backward (elementwise only). `--split auto` runs a micro-batch's DiT pass in the fewest sub-batches whose
   activations fit, so no block is recomputed (exact: per-sample loss weights are unchanged, tested).
4. **FP8 codes kept, dequantised per GEMM** (bit-exactly into a shared scratch buffer), as for the 27B LLM.
5. **Frozen work once per micro-batch.** adaLN modulations of all 34 blocks, the text and latent input projections
   and the rotary tables are computed once under no_grad; the final layer runs on image tokens only.
6. **One flat fused AdamW, one all-reduce, gradient clipping on the device.**
7. **`--compile` of the engine's own elementwise functions only.** Compiling the reference trainer's blocks failed in
   dynamo on its varlen attention path (graph breaks, a guard error; job 4688050); the engine calls attention
   between compiled functions, so attention and the varlen layout never enter dynamo. `--triton_swiglu` reuses perf_kit's SwiGLU
   kernels: inductor's fused kernel for the same expression ran ~5x below memory bandwidth (0.51 ms vs 1.73 ms
   compiled / 2.03 ms eager for forward + backward at 9.1k tokens; forward bitwise, backward within 1 bf16 ulp;
   job 4689171).

One bug in the upstream model matters for training and is handled here: the rotary `inv_freq` is built in float32
as a non-persistent buffer and then cast to bf16 with the model, which rounds it; image tokens sit at position 65,536,
where that rounding becomes phase errors of tens of radians. The loader restores float32 tables and the engine never
runs the rotary under autocast.

**Exactness.** On CPU in float64 (`tests/test_ideogram4.py`) the engine's loss equals the upstream transformer + a
textbook autograd LoRA to 1e-10 relative and every LoRA gradient to < 1e-9, for one adapter, 3 binned adapters and
4 adapters with a base-only region; recompute and split are bit-identical; a saved one-adapter LoRA loads into `peft`
on the upstream transformer and gives the engine's forward. On GH200 with real weights (job 4688444), one 1024 x 1024
micro-batch: loss 0.152538 vs 0.152475 (4.1e-4 relative; the reference against itself with SDPA instead of FA3:
1.3e-4); gradient relative L2 / cosine 0.412 / 0.916 against the reference, while the reference against itself under
a different attention kernel gives 0.404 / 0.919. The engine sits at the reference's own bf16 rounding floor.

**Speed.** One GH200, 4-image micro-batches at 1024 x 1024 (~18.3k packed tokens), the same batches for every arm;
MFU uses the reference trainer's useful-FLOP count (95.16 TFLOP per 1024 x 448 sample = 207.4 MFLOP per pixel, job
4686550) for every row:

| arm | s / micro-batch | MFU | job |
|---|---|---|---|
| reference trainer | 4.09 | 21.5% | 4688444 |
| engine, eager elementwise | 4.17 | 21.1%* | 4688444 |
| engine, eager, every block recomputed | 4.77 | 18.5%* | 4688444 |
| engine, eager + FP8 forward GEMMs (lossy) | 4.13 | 21.3%* | 4688444 |
| engine `--compile` | 2.58 | 34.1% | 4688622 |
| engine `--compile --split auto` | 2.43 | 36.2% (64 GB peak) | 4688622 |
| engine `--compile --split auto --triton_swiglu` | **2.35** | **37.5%** | 4689171 |

\* computed from the reference's count (21.5% x 4.09 s / s). `--compile` measured 2.83 s in 4688444, where another
arm shared the GPU; 2.58 s with the GPU to itself.

Training A/B, one node of 4 GH200, 4 images x 2 accumulation x 4 GPUs = 32 per step, the same batches, timesteps and
noise, 150 steps (median s/step over windows 30-150, validation and sample sheets included):

| arm | s/step | MFU | loss vs reference (window mean rel. diff, mean / max) | job |
|---|---|---|---|---|
| reference trainer | 6.72 | 19.6% | | 4688445 |
| engine `--compile` (split auto) | 3.76 (1.79x) | 34.7% | 6.8e-4 / 1.5e-3 | 4688623 |
| engine `--compile --triton_swiglu` | **3.62 (1.86x)** | **35.9%** | 7.1e-4 / 1.5e-3 | 4689237 |

Per-rank 10-step window losses agree to the 3rd decimal at all 15 windows, and validation loss at fixed t 0.3-0.9 to
<= 1e-4 at steps 50 / 100 / 150 (t 0.1 within its own check-to-check spread). On two nodes with the production run's
layout the engine took 1.85-1.89 s/step against the reference's 3.51-3.53 (1.87x; MFU 33.0% vs 17.0%; jobs 4688868 /
4687262), and the production run then continued on it at 1.79 s/step, 35.5% MFU, to its end at step 9,138 (job
4690086). The whole engine development, benches and A/Bs included, used 7.03 GH200-hours.

**What did not pay.** Eagerly, the engine is no faster than the reference (4.17 vs 4.09 s): its elementwise work cost
~1.3 s per micro-batch uncompiled, and compiling it removes most of that. FP8 forward GEMMs on the checkpoint codes
with per-token activation scales (`--fp8_forward`, row-wise `torch._scaled_mm`) gave no speed-up (4.13 s) and 10x the rounding error (loss 3.9e-3 relative, gradient cosine 0.847
against the exact engine): rejected, as at 27B.

**Released code vs measured code.** `engine.py`, `ops.py` and `kernels.py` are the measured engine; the data / model
loading and the trainer around them were rewritten for this release (no dependence on LogoLabs' trainer or data
formats; images + captions, a plug-in hook, or synthetic data) and are tested on CPU, not yet on GH200. Ideogram 4's
weights are not included; they are licensed under the Ideogram Non-Commercial Model Agreement, and an adapter
trained on them is a Model Derivative under it (`perf_kit/dit/ideogram4/NOTICE.md`).

## 7. The Agate FCDM training loop (`perf_kit/t2i/fcdm`)

*Internal measurements (October 2026). The loop A/B ran on synthetic latents with LogoLabs' private captions and is
not reproducible from this repository; the isolated-step measurement ran on synthetic inputs only (no private data),
but its logs are not in this repository either. Numbers: `results/internal_agate_2026-10.json`, research notes
2026-10-10 / 11.*

![agate_loop](figures/agate_loop.png)

Agate is LogoLabs' small open text-to-image model (Preview 001-002: a 190.9M-parameter generator at 256 px, code and
weights MIT). Its generator is an FCDM U-Net of ConvNeXt-style blocks (7x7 depthwise conv, LayerNorm, per-pixel
modulation, GRN) steered by a recurrent transformer that reads the noisy latent and the text and emits a low-rank
"region map" every block decodes its modulation from. Unlike the LoRA engines, this is full training of a conv net,
and the profile looked nothing like an LLM's: in a 10-step trace (1 GH200, batch 128, compiled) GEMMs and convs were
27% of GPU time and pointwise, reduction and index_put traffic ~60%; the single largest item (16%) was the backward
of the bilinear upsampling every block applied to its 3C modulation maps.

**What changed** (all in `perf_kit/t2i/fcdm`):

1. **Upsample the steering map once (`--steer_upsample_once`).** A 1x1 conv commutes with bilinear interpolation
   (the interpolation weights sum to 1, so the bias passes through), so upsampling each level's r-channel steering
   map once and decoding at full resolution is the same function as decoding at the planner grid and upsampling 3C
   maps in every block, with far less data interpolated. It adds convolution FLOPs and removes memory traffic, so
   MFU is counted with the flag off and the gain judged by images per second. Same function up to float rounding
   (tested on CPU).
2. **EMA as one multi-tensor lerp (`--ema_foreach`).** The same lerp per element as the per-parameter loop, so
   bit-identical (checked on GH200: 579 tensors, max difference 0), in a few fused launches instead of one per
   tensor: EMA host time 24.6 to 1.3 ms per step.
3. **Compile the frozen text encoder's forward (`--compile_text`).** Compiling the bound forward, not the module,
   keeps parameter names (wrapping the module added an `_orig_mod.` prefix and broke the first pilot): text-encode
   host time 47 to 18 ms per step.
4. **DDP `gradient_as_bucket_view`** (and `static_graph` for single-micro-batch steps; with gradient accumulation it
   crashes DDP's reducer, so the trainer refuses that combination).

**Real loop, one node of 4 GH200** (job 3650209; 128 images per GPU, generator compiled in every arm, frozen text
encoder, synthetic 32 x 32 latents with private captions; steady state = median of the 25-step windows 100-225):

| arm (cumulative) | s/step | img/s/GPU | generator MFU |
|---|---|---|---|
| A: as trained | 0.3350 | 382 | 25.1% |
| F1: + EMA foreach, DDP bucket view, DDP static graph | 0.3235 | 396 | 25.9% |
| F2: + steer upsample once | 0.2950 | 433 | 28.4% |
| AG: + GPU-side NaN / spike guard | 0.2960 | 433 | 28.4% |
| H: + compiled frozen text encoder | **0.2700** | **475 (+24%)** | **31.2%** |

A repeat of H (job 3650438) gave 0.2670 s/step, i.e. ~1% run-to-run noise. The GPU-side guard (deciding NaN / spike
skips on the device to drop one host sync per step) changed nothing and is not included. After these changes the step
is GPU-bound (~90% compute busy) and power-capped: SM clock median 1,710 MHz of 1,980, software power cap active in
81% of busy samples.

**The step alone, one GH200** (job 3643558; compiled, synthetic latents and 64-token text context, frozen text;
637.9 GFLOP per image for forward + backward):

| per-GPU batch | as trained | `--steer_upsample_once` |
|---|---|---|
| 128 | 524 img/s, 33.8% MFU | **585 img/s, 37.7%** |
| 192 | 503 img/s, 32.5% | 570 img/s, 36.7% |
| 256 | 504 img/s, 32.5% | 572 img/s, 36.9% |

A batch above 128 per GPU does not raise throughput. With the flag the share of inductor-fused kernels in GPU time
fell from 40.5% to 30.7% and that of GEMMs / convs rose from 36.4% to 42.3%.

**Compile at 512 px (context).** On an internal 512 px LoRA fine-tune of the Agate generator (private data; the LoRA
recipe and the 512 px model are not part of this repository) `torch.compile` took the step from 2.080 to 0.949 s
(2.19x, 62 to 135 img/s/GPU, 61.6 to 34.6 GB) at an equal loss (0.1226 vs 0.1223 at step 120); eagerly, copy and cast
kernels were 43.5% of kernel time and only compile removed them (1.3%). The loop work used 5.50 GH200-hours.

**Released code vs measured code.** `model.py` is Agate's released generator code with the steer-once change;
`ema.py` and `text.py` are the measured EMA and text-encoder code; the trainer is a new, compact front end
(latent shards, a hook or synthetic data; resume, export in Agate's release layout), tested on CPU, not yet on GH200.

## 8. Limitations

- **One machine type.** Every number is from NVIDIA GH200 120GB superchips (96 GB HBM3 per GPU) on one cluster,
  where the module is power-capped at 660 W and the SM clock under load sits at ~1.4-1.8 GHz. We have **no H100
  / A100 / B200 numbers**; x86 hosts dispatch kernels faster than Grace, which helps launch-bound baselines more
  than perf_kit, so the speed-up there is probably smaller. The design is not GH200-specific, the tuning is.
- **LoRA only.** No full fine-tuning, no QLoRA / 4-bit base, no DoRA / rsLoRA, no dropout on the adapter path
  (perf_kit's fused LoRA assumes dropout 0, the setting most LoRA SFT recipes use).
- **Model coverage.** Fast paths exist for the Qwen3.5 hybrid family (Gated DeltaNet + gated attention; tested on
  the 9B and a 27B checkpoint) and for Llama / Mistral / Qwen2 / Qwen3 dense decoders. The dense models are
  CPU-exact and GPU-checked against transformers + PEFT but **not benchmarked on a real checkpoint**.
  Sliding-window attention, MoE layers, vision towers and multi-token-prediction heads are not trained.
- **Text SFT on short rows.** The public benchmark rows average ~530 tokens (single-turn UltraChat). Very long
  sequences shift the balance towards attention / GDN kernels, where perf_kit uses the same kernels the
  baselines can use.
- **Single node.** The benchmark covers 1 GPU and 4 GPUs (one node). perf_kit has run on 2 nodes in production
  (NCCL over Slingshot), but there is no controlled multi-node comparison here.
- **Baselines at their best on this machine, not everywhere.** We tuned micro-batch size and gradient
  checkpointing for each baseline (sweep in the results) and gave them the same fast kernels where they accept
  them (flash-attention 2, Liger, fla's GDN). transformers cannot use padding-free batches for Qwen3.5 here
  without the `causal-conv1d` package (no aarch64 wheel), so its arms are padded (length-grouped); ms-swift runs
  padding-free. Newer releases of any of them may be faster.
- **Short runs.** Throughput is measured over steps 41-60 of 60-step runs, repeated across jobs; loss parity over
  60 steps. That shows the math is the same, not that long runs converge the same. perf_kit has trained
  multi-thousand-step production runs (27B and 9B, private data, up to 2 nodes) at ~36-37% MFU, but those are not
  part of this benchmark.
- **Energy.** Measured per job with perun (GPU module + NIC); this cluster has no independent energy meter to
  cross-check against.
- **Ideogram 4 and Agate.** Their numbers (sections 6-7) are internal: private data, or synthetic inputs whose logs
  are not in this repository, and LogoLabs' internal trainers around the released engine code. The released trainers
  are tested on CPU only; there is no public, reproducible benchmark for them yet.

## 9. How to reproduce

Everything below runs from this repository; the only external inputs are the model weights and the public data. (It
covers the LLM benchmark; sections 6-7 have no public benchmark.)

1. **Data** (login node, needs internet): download the two parquet files of
   [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) (MIT) at
   revision `8049631c405ae6576f93f445c6b8166f76f5505a` and build the single-turn set:

   ```bash
   hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --revision 8049631c405ae6576f93f445c6b8166f76f5505a \
       --include "data/train_sft-00000-of-*.parquet" "data/test_sft-00000-of-*.parquet" --local-dir src
   python bench/data/prepare_ultrachat.py --src src --out data --revision 8049631c405ae6576f93f445c6b8166f76f5505a
   ```

   This must reproduce `train.jsonl` sha256 `4e87fafa...7b68d` and `val.jsonl` sha256 `9a719e05...4f69`
   (full hashes in `results/data_manifest.json`).
2. **Weights**: `Qwen/Qwen3.5-9B-Base` (Apache-2.0) in a local directory (compute nodes may be offline).
3. **Site file**: copy `bench/slurm/site_env.example.sh`, fill in your modules / venvs / library dirs.
4. **Run** (one node; the account and partition are yours). The exact arm lists we ran:

   ```bash
   export SITE_ENV=$PWD/site_env.sh REPO=$PWD MODEL=<9B dir> DATA=<data dir> RUNS=<output dir> STAGE_MODEL=1
   L1="--nodes 1 --ntasks-per-node 1 --gpus-per-node 1 --cpus-per-task 72"
   L4="--nodes 1 --ntasks-per-node 4 --gpus-per-node 4 --cpus-per-task 72"
   ARMS="pk@12288 hf@8 hfx@16 hfx@4+nockpt swift@8+nockpt"
   sbatch -A <acct> -p <part> $L1 --export=ALL,TAG=main,ARMS="tests $ARMS unsloth@8" bench/slurm/bench_arms.sbatch
   sbatch -A <acct> -p <part> $L4 --export=ALL,TAG=node,ARMS="$ARMS pk@12288" bench/slurm/bench_arms.sbatch
   sbatch -A <acct> -p <part> $L1 --export=ALL,TAG=warm,ARMS="pk@12288 pk@12288 pk@12288" bench/slurm/bench_arms.sbatch
   sbatch -A <acct> -p <part> $L1 --export=ALL,TAG=ablate,ARMS=ablate,ABLATE="mt8192 mt3072 mt4096 mt6144 mt10240        mt12288 mt16384 old_planner_mt8192 old_planner_mt12288 ckpt_all no_liger no_fla_conv no_swiglu_kernel        no_fused_gate no_fla_gnorm no_flash_attn no_fused_grad_acc fa3 flashqla mt8192" bench/slurm/bench_arms.sbatch
   # 27B: MODEL=<Qwen3.8-27B-FP8 dir> PK_STORAGE=fp8 STEPS=40 ARMS="pk@6144 pk@3600"
   ```

   An arm token is `<framework>[@<micro>][+nockpt]` (micro = rows per micro-batch, or tokens for perf_kit); the same
   token twice in one job gives `_r2`, `_r3` repeats (perf_kit repeats run with a warm kernel cache).
5. **Analyse**: copy each `<RUNS>/<tag>_<jobid>` directory to `results/raw/`, then

   ```bash
   python bench/analysis/aggregate.py --raw results/raw --out results
   python bench/analysis/plots.py --summary results/summary.json --out figures
   ```

Without Slurm, run the arms directly: `python run_train.py ...` / `python bench/e2e/hf_peft_train.py ...` with the
same flags as in `bench/slurm/bench_arms.sbatch` (`torchrun` works for multi-GPU; set `--step_order sequential`
for perf_kit so all frameworks see the same step sets).

## 10. Advanced mode: growable GPU pool (`perf_kit/pool`)

**Problem.** On Arrhenius (nodes of 4 GH200, ~1,400 jobs pending in the `gpu` partition on 2026-11-24) a whole node
can wait a long time while 1-2 GPU jobs start in seconds to minutes. Synchronous DDP needs all its GPUs at once.
The pool is an opt-in, advanced mode, intentionally designed to maximise the use of GPUs that are only partly used:
the free GPUs left on partly allocated nodes and the short free windows of a busy cluster. It is not a replacement
for DDP when whole nodes start quickly; it measured worse on short runs, and with untuned outer settings it
failed on a 104 M-parameter conv VAE (both below).

**Method (published, adapted).** DiLoCo (Douillard et al. 2023, [arXiv 2311.08105](https://arxiv.org/abs/2311.08105)):
each worker trains H inner steps (AdamW) from the global weights; the mean pseudo-gradient (global minus local) is
the gradient of an outer Nesterov SGD (lr 0.7, momentum 0.9 in the paper; their adaptive-compute experiment found
quality depends on total compute, not on how it is spread over time, and dropping up to 50% of outer gradients cost
2.1% perplexity). Workers here arrive asynchronously, so the outer step is Liu et al.'s Delayed Nesterov
("Asynchronous Local-SGD Training for Language Modeling", 2024, [arXiv 2401.09135](https://arxiv.org/abs/2401.09135),
Algorithm 3 with c = 0: each arrival moves the weights by lr x its share, the momentum is updated once a full
round has arrived), with two adaptations: shares are **sample-weighted** (FedAvg's n_k / n, McMahan et al. 2017,
[arXiv 1602.05629](https://arxiv.org/abs/1602.05629)), and a round is the samples the *currently publishing*
workers produce per round, so a 2-GPU job counts as two 1-GPU jobs and a hung job stops counting. Staleness:
contributions more than 4 rounds stale are dropped; the `(1 + s)^-0.5` scaling is available but off, because Liu
et al. report it gave only "marginal benefits" on top of Delayed Nesterov (FedBuff's buffered form, Nguyen et al.
2022, [arXiv 2106.06639](https://arxiv.org/abs/2106.06639), is the `mode="buffered"` option). Transport is the
shared filesystem: immutable version files claimed with `os.link` (create-if-absent), an O_EXCL lock with stale
takeover, per-worker high-water marks so a crash between commit and inbox clean-up never applies a contribution
twice. Not built: overlapping the sync with compute and quantised deltas (Streaming DiLoCo, Douillard et al. 2025,
[arXiv 2501.18512](https://arxiv.org/abs/2501.18512)), and the `torchrun --nnodes=min:max` elastic mode.

**One finding from the cluster runs.** The stored global weights are the Nesterov look-ahead point; evaluated
directly they swung by up to 0.4 bits/byte between consecutive versions of one run (1.66, 1.40, 1.76 bpb for
versions 24-26 of repeat 2, `bench/pool/eval_versions.py`, job 5713885). The same versions with the look-ahead
removed (x = theta + lr x momentum x m, which for the synchronous update is exactly the heavy-ball iterate; tested
against `torch.optim.SGD(nesterov=False)`) moved smoothly (1.33, 1.32, 1.36). `Pool.load_global` therefore returns x;
workers still train from the stored point, as in DiLoCo. This rule was adopted after seeing repeat 2, so r1-r3 below
are re-evaluations of their final versions and r4 is the only repeat run with it fixed in advance; at momentum 0.5
(r5) the stored point was slightly better than x (1.391 vs 1.398), so x is not better everywhere.

**Comparison.** Byte-level GPT (12 layers, d 768, 85 M parameters, context 256) on UltraChat single-turn text
(`perf_kit/pool/example_lm.py`), AdamW lr 1e-3, 5% warm-up, cosine to 0.1x, 64 windows per GPU per step, 2,048
fixed validation windows, bits per byte. DDP: one job of 4 GPUs (global batch 256). Pool: four jobs submitted
together with earliest start times 0 / 60 / 120 / 180 s: 1 GPU, 2 GPUs, 1 GPU (hard-killed 60 s into training,
`--die_after_s`, mid-round), 2 GPUs; H = 500; outer lr 0.7, momentum 0.9 (not tuned). Same total samples
(1.2 M windows = 307 M bytes, ~6.8 passes over the 45 MB training text, so the regime overfits: DDP's final training
loss is ~0.95 against 1.33 validation). `bench/pool/compare.sh` + `bench/pool/summarize.py`; times from sacct.
These runs drove the pool's learning-rate schedule by `Pool.progress` (`--schedule_by progress`, still the default
of `compare.sh` so the table reproduces), before `progress_est` existed; see the schedule pitfall below.

| run (1.2 M samples) | val bpb (pool: exported x) | pool: stored look-ahead | GPU-h | busy / idle / sync GPU-h | wall start to end | queue wait (from eligible) | samples trained / applied |
|---|---|---|---|---|---|---|---|
| DDP 4 GPUs, r1 (job 5712506) | 1.3327 | | 0.279 | 0.221 / 0.058 / 0 | 251 s | 41 s | 1.20 M / 1.20 M |
| DDP 4 GPUs, r2 (5713256) | 1.3350 | | 0.290 | 0.223 / 0.067 / 0 | 261 s | 642 s | 1.20 M / 1.20 M |
| pool 1+2+1+2, r1 (5712507-10) | 1.2955 | 1.3350 | 0.323 | 0.249 / 0.074 / 0.010 | 382 s | 41, 161, 101, 101 s | 1.34 M / 1.22 M |
| pool 1+2+1+2, r2 (5713257-60) | 1.3626 | 1.7634 | 0.301 | 0.242 / 0.060 / 0.011 | 288 s | 28, 27, 5, 5 s | 1.30 M / 1.22 M |
| pool 1+2+1+2, r4 (5714236-39) | 1.2993 | 1.3396 | 0.293 | 0.236 / 0.057 / 0.010 | 320 s | 12, 72, 12, 12 s | 1.30 M / 1.25 M |

| run (400 k samples, H = 250, starts 0 / 30 / 60 / 90 s, kill after 40 s) | val bpb | look-ahead | GPU-h | wall | queue wait |
|---|---|---|---|---|---|
| DDP 4 GPUs, r3 (5713261) | 1.3645 | | 0.142 | 128 s | 1,002 s |
| pool 1+2+1+2, r3 (5713262-65) | 1.5766 | 1.7878 | 0.134 | 150 s | 28, 58, 28, 35 s |
| pool 1+2+1+2, r5, outer momentum 0.5 (5715039-42) | 1.3975 | 1.3911 | 0.165 | 204 s | 54, 144, 54, 84 s |

What this shows, and does not:

- **Quality at 1.2 M samples: no loss measured.** Pool 1.296 / 1.363 / 1.299 bpb (n = 3, mean 1.319) against DDP
  1.333 / 1.335 (n = 2), with one of the four pool jobs killed mid-round in every repeat. The pool's spread (0.07) is
  larger than DDP's (0.002); n = 3 cannot say whether its lower mean is real. Values for r1 and r2 are the exported x
  of their final version (evaluated after the runs by job 5713885); r4 ran the final code, which reports x itself.
- **Short runs lose at momentum 0.9.** At 400 k samples (6 completed outer rounds) the pool is clearly worse (1.577 vs
  1.365, n = 1). Hypothesis, not proven: final quality tracks completed outer rounds (13, 8, 6 rounds gave 1.296,
  1.363, 1.577), and momentum 0.9 remembers ~10 rounds, the whole of a short run. One run at momentum 0.5 (r5) got
  1.398, consistent with it, but its job mix differed (the 2-GPU job started 144 s late, so the 1-GPU job did more
  rounds alone). H, outer lr and momentum were not tuned.
- **Cost: +7% GPU-hours** (0.306 vs 0.284 mean at 1.2 M): every job pays its own start-up (imports, data load, NCCL
  init) and final evaluation, 16-64 s per job; the sync itself (publish, apply, pull of 85 M fp32 weights over
  Lustre, 0.7-0.9 s each) is ~4% of busy time at H = 500; 4-9% of trained samples are never applied (the killed job's
  partial round, rounds in flight when the run finishes; r1 ran before the in-round DONE check and lost the most).
- **A killed job keeps its share for up to `heartbeat_timeout` (180 s).** In every repeat the round size stayed
  192 k samples (all four jobs) until the end, because the killed job's heartbeat had not yet expired: the
  survivors' contributions were scaled by 160/192 for the rest of these short runs. What the runs show is that
  nothing waits for a dead job, not that it is evicted quickly (eviction itself is tested on CPU).
- **Queue.** On 2026-11-24 the full node waited 41 s, 642 s and 1,002 s (the scheduler had estimated ~2.5 h for the
  last two); the pool's jobs started 5-161 s after becoming eligible. Time from first eligibility to the final model:
  DDP 292 / 903 / 1,130 s, pool 423 / 316 / 332 / 178 s. The pool's own wall time is longer because it starts with
  one GPU.
- **Not measured against DDP on this model:** larger models (sync cost grows with size; the file I/O was not
  optimised), pools of more than 6 GPUs, a multi-node job joining, runs of hours, outer-lr tuning. The one larger
  pool tried (24 GPUs, the real use below) failed against DDP.

**Schedule pitfall (found in real use, fixed).** `Pool.progress` is the applied samples at the worker's last pull
plus its own unapplied samples. Until a worker's first outer sync nothing of the other workers is applied, so a
schedule driven by it runs the first round of a fresh run (and a late joiner's first round) as if the pool were
this one job: in the real run below, with a progress-based warm-up, the first 500 local steps trained at
~3% of the nominal learning rate. `Pool.progress_est` (new; `progress` keeps its meaning) scales the worker's
unapplied samples by R / r, where R is the pool's live round size (the other workers publishing on time, from their
heartbeats, refreshed at every pull and every `done_check_every` steps and broadcast to every rank, plus this
worker) and r this worker's own samples per round (from `step_samples`, or measured from its pace). It assumes every
live worker is as far into its round as this one, is never below `progress`, never decreases (it holds its
end-of-round value until the applied samples catch up), and equals `progress` for a pool of one; `done` still
depends only on applied samples. A worker that joined without `step_samples` beats a round size of 0 until its
first publish, so the others cannot count it during exactly the round this is for: pass `step_samples`. Tested on
CPU, including that both ranks of a 2-rank job use the same learning rates. Each joining job's inner AdamW also
starts with empty moments, so warm up by local steps as well (`Pool.local_steps`; `example_lm.py --local_warmup N`).

**Real use: a failure (2026-11-25).** A 104 M-parameter convolutional image VAE (GroupNorm) at 1024 px trained as a
pool of 12 x 1-GPU + 6 x 2-GPU Slurm jobs (24 GPUs) on GH200 nodes that were only partly used, with untuned outer
settings (outer lr 0.7, momentum 0.9, H = 500, asynchronous Delayed Nesterov, schedule by `progress`). Throughput and
queue worked: at 03:00 CET about 87 free GPUs sat on partly allocated nodes while whole-node jobs waited hours; the
small jobs started within minutes (a 2-job smoke test within one second), trained at ~19.6 samples/s per GPU, and 0
contributions were dropped in the first 67 global versions. Quality did not: the run was stopped as a failure at ~22%
of its length (1.34 M samples, 145 global versions, 22.4 GPU-hours). On the exported global weights (`load_global`,
the heavy-ball iterate) held-out glyphs had a worst-point median error of 2.66 px (512 px scale) with 5.9% within 5
units, and logo layers 2.40 px, 8.2% and overlap 0.869; a 2-GPU synchronous DDP run from the same start reached ~1.0
px and 95-99% after only 48 k samples. The stored look-ahead point was worse still (3.8 px, 0%), and the in-run
validation, taken right after a pull and so on the look-ahead point, read 5.9 px / 0%. Meanwhile each worker's own
training loss was low (reconstruction ~0.015, against 0.06-0.13 for the global weights): the averaged model was much
worse than the models it averaged. The cause is not established. Candidates: outer momentum 0.9 with many
asynchronous workers; H = 500 with a fresh inner AdamW in every job; the slow progress-based warm-up of the first
rounds (the pitfall above); and GroupNorm conv nets may average worse than transformers. Its trainer used the pool
together with perf_kit's loop optimisations: `torch.compile` of the encoder and decoder, no host syncs in the step,
data rendered ahead on a side CUDA stream, recompute of one branch to fit memory, `channels_last`, fused AdamW and
DDP gradients as bucket views inside each job; data seeds included the worker index (`Pool.index`) and validation ran
in whichever job won `Pool.claim(f"val_{k}")`. The pool adds nothing to the step: between outer syncs `Pool.sync`
only counts, and every `done_check_every` steps rank 0 checks the DONE file and reads the live jobs' heartbeats (one
small JSON file per job), then broadcasts the result. Lesson, now in the README: validate the exported global weights
early against a short DDP run from the same start before committing GPU-hours; a worker's own training loss says
nothing about the averaged model.

Compute for the comparison (smoke tests, the hung first smoke, the repeats, the evaluation job): 2.29 GPU-hours
(sacct); the failed real-use run above: 22.4 GPU-hours.

### 10.1 Why the real-use run failed: outer momentum with stale contributions

The night's own logs already show it. Median reconstruction loss over the 18 workers (`rec_avg`, mean of 50 steps)
around each outer sync (one sync every 500 local steps):

| round | workers | 50 steps before the sync | first 50 steps after the pull | 51-100 after | 201-250 after |
|---|---|---|---|---|---|
| 1 | 18 | 0.2818 | 0.2003 | 0.0850 | 0.0239 |
| 2 | 18 | 0.0148 | 0.1229 | 0.0243 | 0.0151 |
| 3 | 18 | 0.0128 | 0.0592 | 0.0169 | 0.0138 |
| 4 | 18 | 0.0135 | 0.0487 | 0.0162 | 0.0135 |
| 5 | 18 | 0.0127 | 0.0461 | 0.0157 | 0.0131 |
| 6 | 14 | 0.0128 | 0.0702 | 0.0173 | 0.0135 |
| 7 | 13 | 0.0128 | 0.0609 | 0.0150 | 0.0131 |
| 8 | 13 | 0.0118 | 0.0306 | 0.0129 | 0.0119 |
| 9 | 12 | 0.0111 | 0.0520 | 0.0128 | 0.0110 |
| 10 | 12 | 0.0109 | 0.0577 | 0.0125 | 0.0101 |

Every pull lands the workers on weights 3-10x worse than the ones they left, and each worker repairs them within
~100 steps: training loss, so the stored global weights themselves are bad and the evaluation path is not to blame.

| hypothesis | test | result |
|---|---|---|
| different initial weights per worker | 3 processes built from seeds 1000 / 1001 / 1002, then their first pull | ruled out: all pulled version 0 with the same sha256; now asserted |
| buffers not pooled | the VAE's buffers | ruled out: none (GroupNorm only); the pool now logs any frozen parameter or non-float buffer it does not pool |
| `_orig_mod.` / `module.` name mismatch, partial copy | `_pull` indexes every local parameter | ruled out; the pool now raises at the first pull if the name sets differ in either direction |
| bad load at evaluation | `load_global` against a fresh model loaded from the same version | ruled out: identical loss (< 1e-6) |
| outer settings | the quadratic and tiny-VAE grids below | **reproduced, and fixed by momentum 0** |

**Quadratic model through the pool's own `outer.apply`** (no training; each worker travels a fraction h of the way to
its local optimum in a round, contributions about one round stale as in the run; distance to the optimum after 60
rounds, start 1; lr 0.7, momentum 0.9):

| workers | h = 0.3 | h = 0.6 | h = 1.0 |
|---|---|---|---|
| 2 | 0.0005 | 7e-6 | 4e-6 |
| 3 | 0.0005 | 9e-5 | 0.004 |
| 4 | 0.0002 | 0.002 | 0.29 (oscillates) |
| 5 | 0.0003 | 0.011 | 1.0 (diverges) |
| 8 | 0.001 | 0.086 | 7.2 |
| 18 | 0.0025 | 0.42 | 64 |
| 24 | 0.0027 | 0.56 | 73 |

With momentum 0, staleness s (in rounds) is what remains; distance after 60 rounds, 24 workers:

| lr, h | s = 1 | s = 1.5 | s = 2 | s = 2.5 | s = 3 |
|---|---|---|---|---|---|
| lr 1.0, h 1 | 6e-6 | 0.47 | **34** | **240** | **710** |
| lr 0.7, h 1 | 4e-10 | 2e-4 | 0.17 | **1.3** | **7.1** |
| lr 0.5, h 1 | 4e-14 | 8e-7 | 3e-4 | 0.008 | 0.46 |

**Tiny version of the real VAE on CPU** (the trainer's UVAE class and loss at widths 16 / 32 and 64 px, warm-started,
24 simulated workers unless noted; pulled / local = loss of the pulled global weights over the workers' own):

| arm | rounds | local | pulled (global) | pulled / local, 2nd half (median, max) |
|---|---|---|---|---|
| 24 workers, lr 0.7 / mom 0.9, async (as run) | 15 | 0.0366 | **0.0544** | **1.46, 1.71** |
| 24 workers, lr 0.7 / mom 0, async | 15 | 0.0409 | 0.0353 | 0.86, 0.87 |
| 24 workers, lr 1.0 / mom 0, async | 15 | 0.0402 | 0.0348 | 0.87, 0.87 |
| 24 workers, lr 0.7 / mom 0.9, synchronous | 15 | 0.0362 | 0.0317 | 0.86, 0.88 |
| 3 workers, lr 0.7 / mom 0.9, async | 60 | 0.0294 | 0.0314 | 1.11, 1.23 |
| 3 workers, lr 1.0 / mom 0, async | 60 | 0.0354 | 0.0327 | 0.92, 0.96 |

The 3-worker rows are why a 2-3 job smoke test cannot catch this. The tiny model reproduces the pattern (an
oscillating global, healthy workers) but not the size (1.5x here, 3.6-8x on the cluster).

**What changed in `perf_kit/pool`.**
- A warning at start-up when outer momentum is above 0 in asynchronous mode (the default stays 0.9 so section 10's
  comparison reproduces; use 0 for more than ~3 asynchronous workers, lr 0.7 tolerates contributions up to ~2 rounds
  stale, lr 1.0 only ~1.5).
- Health probe: `Pool(..., probe=fn)` with `fn(model) -> float` on a fixed batch; at every sync the loss of the local
  and of the pulled weights and their ratio go to `<run>/health/<worker>.jsonl`, and when the median ratio of the
  last 5 syncs (at least 3) exceeds `health_warn` (1.25) it is logged and written to `events.jsonl` once. A
  "3 in a row" rule missed the oscillating case, hence the window. On replay it fires at round 7 of the as-run
  tiny-VAE arm and at the third sync of the night's run, and stays silent on every healthy arm.
- `first_round_s`: a live worker that has not published yet counts in the round size (the first round took ~800 s
  against a 300 s publish window, so its first 12 contributions had counted as 2.67 rounds instead of 1).
- First-pull checks: the same parameter names on both sides and the hash of the pulled version.

CPU integration test of the trainer's pool path (3 processes from different seeds, 40 local steps per round, 12 k
samples): with momentum 0 the probe ratio is 0.97 median (1.00 max at lr 0.7) and no alarm; with the as-run 0.7 /
0.9 it is 1.16-1.17 median and the alarm fires.

**GPU proof** (Arrhenius GH200, VAE at 512 px warm-started from the same checkpoint, 144 k samples each):

| arm | jobs | held-out glyphs within 5 u | worst-point median (px, 512 scale) | pulled / local at the last sync | wall |
|---|---|---|---|---|---|
| pool: 3 x 1-GPU jobs, lr 1.0, momentum 0, H = 100, health probe | 3 jobs joined within 0.1 s | **0.906** | 0.591 | 0.97-0.98 | ~30 min |
| plain: one 1-GPU job, batch 96 | 1 | 0.902 | 0.592 | - | 67 min |

GPU-hours: 2.6 (sacct). The proof has 3 workers, which would not have shown the old failure; the claim for ~24
workers rests on the simulations above.

## 11. Advanced mode: elastic synchronous DDP over TCP (`perf_kit/elastic`)

The pool keeps jobs loosely coupled and pays for it with drift between syncs and with outer-optimizer settings that
must be right. If the jobs can reach each other fast enough to all-reduce every step, the run can instead keep the
exact math of one big DDP job while still being made of small jobs that start within minutes.

### 11.1 The network between separate jobs

Arrhenius uses HPE Slingshot (`SwitchType=switch/hpe_slingshot`, libfabric `cxi`, aws-ofi-nccl 1.19.2). Inside one
job, NCCL runs over cxi through the OFI plugin. Between jobs:

| test (2026-11-25) | result |
|---|---|
| one job, 4 x 1 GPU on 4 nodes (`--nodes=4 --gpus-per-node=1`) | refused at submission: "You need to allocate at least (nodes-1)*4+1 GPUs" (site rule: multi-node jobs take whole nodes) |
| one job, `--ntasks=4 --gpus=4` without a node count | accepted by the filter, but would only get past the rule by not stating its nodes: cancelled before it started |
| fast path between separate jobs | Slurm gives each job its own VNI (`SwitchParameters ... job_vni=user`); not attempted |
| plain IP over `hsn0` between two separate 1-GPU jobs on different nodes | works |

Two separate 1-GPU jobs on different nodes, rendezvous through a file on the shared filesystem:

| measurement | time | bandwidth |
|---|---|---|
| raw TCP, one stream, 1 GiB | - | 2.9-3.5 GB/s |
| gloo all-reduce on CPU, 160 MB | 107-124 ms | 1.4-1.6 GB/s |
| NCCL socket transport, GPU, 16 MB bf16 | 2.5 ms | 6.6 GB/s |
| NCCL socket transport, GPU, 160 MB bf16 | **21.6 ms** | **7.8 GB/s** |
| NCCL socket transport, GPU, 512 MB bf16 | 72.5 ms | 7.4 GB/s |
| all four `hsn` interfaces, `NCCL_NSOCKS_PERTHREAD=4`, `NCCL_SOCKET_NTHREADS=4` | hung on the 160 MB collective until the job's time limit | - |

### 11.2 Literature and what was adapted (from memory unless marked)

| idea | source | used here as |
|---|---|---|
| membership change = restart every worker from a checkpoint | TorchElastic / torchrun rendezvous (PyTorch docs) | rejected as too slow for jobs that come and go every few minutes; kept: a rendezvous through a shared store and a checkpoint for a full restart |
| in-memory state, commit / restore at step boundaries | Elastic Horovod (Horovod 0.20 docs, 2020) | a step is committed only after the gradient sync is verified; membership changes happen at step boundaries |
| per-step quorum, a healthy peer sends its live state to the newcomer, abortable process groups | torchft (pytorch/torchft, 2024-25) | the epoch's source rank broadcasts model, optimizer and counters; the group is aborted and rebuilt in-process |
| gradient buckets all-reduced during backward, `no_sync` for accumulation | Li et al., VLDB 2020 (PyTorch DDP) | plain DDP inside an epoch, rebuilt per epoch |
| bf16 gradient compression | `ddp_comm_hooks.default_hooks.bf16_compress_hook` (checked in torch 2.13) | on by default on the GPU (NCCL only) |
| linear / square-root lr scaling | Goyal et al. 2017; Malladi et al., NeurIPS 2022 | `lr_rule="linear"` / `"sqrt"` (default) |
| a fixed global batch while the GPU count changes | Pollux (Qiao et al., OSDI 2021) | `global_micro=G` |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` modes, `TORCH_NCCL_BLOCKING_WAIT`, `_abort_process_group`, DDP `init_sync` | read in torch 2.13's sources | verified |

### 11.3 Design

| file in the run folder | what |
|---|---|
| `members/<job>-<local rank>-<pid>.json` | IP on the interface, job id, job size, state (joining / active / recovering), epoch, committed step, heartbeat every 5 s |
| `epochs/eNNNNNN.claim` | O_EXCL: whoever creates it leads that epoch (exactly one winner, also when every survivor races after a crash) |
| `epochs/eNNNNNN.json` | the roster: members in rank order, rank 0's IP and port (the epoch's TCPStore), the source rank, the reason |
| `leave/<job id>`, `STOP`, `DONE` | a job's TERM trap touches its leave file; STOP ends the run; DONE is written by the last rank 0 |
| `ckpt/step_*.pt`, `metrics/elastic.jsonl`, `events.jsonl` | resume checkpoint, per-step health, one line per epoch / failure / join / leave / checkpoint |

One optimizer step:
1. Backward on this rank's micro-batches; DDP all-reduces the buckets (bf16 hook), `no_sync` on every micro-batch but
   the last.
2. A small control all-reduce, waited with the timeout: alive count, join pending, stop, superseded; one agreed value
   from rank 0 (`agree()`, e.g. the time fraction of a deadline schedule); each rank's loss and leave flag; a parameter
   checksum per rank every 50 steps. NCCL runs collectives in order, so a correct alive count proves that the gradient
   sync finished on every rank.
3. Only then `optimizer.step()`. If anything failed before this point, weights and optimizer are untouched: the step is
   discarded and redone in the next epoch on the same micro-batch indices.
4. Join, leave, stop and desync are read from the same control vector, so every rank takes the same branch.

Membership changes:
- **Join:** a job announces itself only after its own set-up. Rank 0's watcher thread sees it, and at the next step
  boundary the leader writes epoch n+1 (survivors first, then newcomers grouped by job, so NVLink neighbours are
  adjacent); everyone destroys the old group and initialises the new one; the source rank broadcasts the counters and
  the extra dict, then all tensors (model, extra modules such as the EMA, optimizer state) in flat per-dtype buffers;
  every rank checks a sha256 of parameters and buffers against the source's; DDP is rebuilt with `init_sync=False`.
- **Leave:** Slurm's `--signal=B:TERM@120` reaches the batch shell, whose trap touches the leave file; a second,
  signal-free trigger is `ELASTIC_LEAVE_AT` = the job's end time from `squeue %e` minus a margin. The job finishes its
  step and the rest regroup. If every rank leaves, rank 0 checkpoints first and the next job resumes from it.
- **Crash / hang:** the step's collective raises (under 1 s for a dead peer, at the timeout for a hung one). The
  survivors abort the group and check in as recovering with their committed step; the first to claim the next epoch
  waits until every old member has checked in or is dead (heartbeat older than 20 s); everyone receives the state of
  the survivor with the highest step (survivors can be one step apart, so the whole state goes to everyone). A hung
  process that wakes later finds it is not in the roster and comes back as a newcomer.
- **Backstops:** a stall watchdog (a rank with no committed step for 3 timeouts plus the gather time, outside a hold,
  exits with code 75; the gloo case, where a DDP all-reduce with a vanished peer never returns, and a deadlocked data
  loader); a split-brain guard (a group that finds a newer roster without itself stops and rejoins); `hold()` for long
  rank-0 work, during which the other ranks wait on the TCPStore instead of in a collective.
- **Trap:** a site default of `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` tears the process down on any NCCL error, which kills
  the survivors; the job script sets 2 (abort the communicator, keep the process) and `TORCH_NCCL_BLOCKING_WAIT=1`.

### 11.4 GPU probe: does NCCL survive a lost peer in-process?

3 separate 1-GPU jobs on 3 nodes, 6.8 GPU-minutes:

| measurement | value |
|---|---|
| NCCL init, 3 jobs, first epoch | 1.04 s |
| planned regroup (destroy + rendezvous + init), 3 times | 0.86-1.80 s |
| small all-reduce (16 floats), world 3 | 1.23 ms |
| 160 MB bf16 all-reduce, world 3 | 42 ms |
| 1 GB fp32 broadcast | 123 ms (8.1 GB/s) |
| peer killed (`os._exit`) mid all-reduce: detected | 0.58 s, `DistBackendError ... Connection closed by remote peer` |
| then abort + regroup with world 2 | 1.25 s |
| peer hung 70 s: the waiting rank | timeout at 45.5 s, then abort; regroup when the sleeper woke |

### 11.5 GPU proof against a single-GPU run at equal samples

Byte-level GPT (12 layers, d 768, 85 M parameters, context 256) on UltraChat single-turn text; AdamW lr 1e-3, 5%
warm-up, cosine to 0.1; 2,000 steps x 384 windows = 768 k samples; fixed global batch of 12 micro-batches x 32; bits
per byte on 2,048 validation windows. Reference: the same script (`--mode ref`) on one GPU accumulating the same 12
micro-batches.

Run p1: A, B and C started together on 3 nodes; C hard-died in its 600th step; B had a 6 min limit, so Slurm's TERM
came at T-120 s; D had an earliest start 150 s later.

| epoch | world | what happened | regroup (s) | NCCL init (s) | state transfer (s) | at step |
|---|---|---|---|---|---|---|
| 0 | 3 | founded (A, B, C on 3 nodes) | 9.9 | 8.2 | 0.08 | 0 |
| 1 | 2 | C died: recovery (17.5 s waiting for C's heartbeat to expire) | 18.5 | 0.67 | 0.05 | 600 |
| 2 | 1 | B left on Slurm's TERM | 0.51 | 0.03 | 0.01 | 1,024 |
| 3 | 2 | D joined (a fourth node) | 1.40 | 0.64 | 0.05 | 1,137 |

| run | final val bpb | train loss difference to ref 1, steps 1-200 / 201-600 / 601-1200 / 1201-2000 |
|---|---|---|
| ref 1 (1 GPU) | 1.3134 | - |
| ref 2 (1 GPU, repeat) | 1.3142 | -0.0038 / -0.0029 / -0.0005 / -0.0005 |
| elastic p1 (3 -> 2 -> 1 -> 2, bf16 sync) | **1.3081** | -0.0125 / -0.0131 / -0.0025 / -0.0018 |
| elastic p2 (2 jobs, fixed world, fp32 sync, both on one node) | 1.3093 | -0.0087 / -0.0081 / -0.0022 / -0.0026 |

- **Same math:** the first logged steps (10, 20, 30) agree with the reference to 1e-4 in loss; a wrong loss scale, lr
  or batch would show there at once. Afterwards the trajectories drift apart chaotically, as the two references also
  do (the reference itself is not bit-reproducible on the GPU).
- **Not explained:** both elastic runs end ~0.005 bpb lower than both references (0.4%, against spreads of 0.0008
  between the references and 0.0012 between the elastic runs). With n = 2 per arm this cannot be told from luck; no
  code difference that would favour elastic is known, and it is not the bf16 hook (the fp32 run is lower too).
- **Same node:** in p2 both jobs landed on one node and talked over sockets, because `NCCL_HOSTID` differs per job.

Speed (median per step; the reference takes 21.3 ms per micro-batch):

| world | step (ms) | compute share, 12 / W micro-batches (ms) | sync overhead per step (ms) | control wait (ms) | speed-up vs 1 GPU |
|---|---|---|---|---|---|
| 1 | 266 | 256 | 10 | 0.15 | 0.96 |
| 2, bf16 sync | 154 | 128 | 26 | 1.3 | 1.66 |
| 3, bf16 sync | 116 | 85 | 30 | 1.0 | 2.21 |
| 2, fp32 sync (same node) | 175 | 137 | 38 | 0.5 | 1.57 |

With an 85 M model synced every 4-6 micro-batches, TCP between jobs costs 26-30 ms per step, about 20-26% of the
step at 2-3 GPUs; the bf16 hook saves about a third against fp32. A model with longer steps pays relatively less.

### 11.6 Real use: an overnight font generator (2026-11-26)

An 80 M-parameter flow-matching generator over the latents of a silhouette VAE, with a style-encoder branch, trained
as a job array of 16 x 1-GPU jobs of 5:45 h, at most 8 at once, staggered leave times, a run deadline 11 h after
creation. Per-GPU batch 128, lr 2e-4 x sqrt(128 W / 512), cosine over samples or time to the deadline (rank 0's clock
through `agree()`), EMA updated per committed step through `post_commit` and transferred to newcomers, evaluations
inside `hold()`. A whole 4-GPU node was estimated to start ~14.5 h later; the 1-GPU jobs started within minutes.

| item | value |
|---|---|
| world | 2 -> 3 -> 4 -> 6 -> 8 as jobs started |
| desync / recovery / stall events | 0 / 0 / 0 |
| parameter checksums (model + EMA) | all equal (37 checks, 0 different, at step 3,700) |
| step time at world 8 (1,024 samples) | 0.49-0.58 s, about 80-85% of 8 independent GPUs |
| per-rank train loss at step 3,700 | 0.12-0.16 (different data, same weights) |
| validation loss, conditioned on the style vector | 0.2730 at 1.02 M samples, 0.1774 at 2.56 M |
| evaluation cost | ~300 s with sample sheets, the other ranks idle (~10% of the time) |

Two bugs were found by the first launch and fixed before the run above (15 min, 1.9 GPU-hours): the EMA was updated
after `step()`, so a newcomer missed one update (now `step(post_commit=...)`, with a test); and every rank refreshed
its latent pool on its own counter, so almost every synchronous step waited for one rank's refresh (0.90 s per step
at world 8; now on the global step, every 4 steps).

### 11.7 Tests and limits

`tests/test_elastic.py` (CPU, gloo, real local processes): micro-batch split and lr rule for W = 1..12; one winner
among 5 members racing for an epoch claim; optimizer state transferred bit-identically into a fresh AdamW; a fixed
world of 3 equal to the 1-process reference (loss curves equal to float noise, equal checksums, final validation
within 5e-3); worlds 1, 2, 3, 2, 3, 2 through join, join, leave, crash (`os._exit`) and late join with a maximum loss
difference of 4e-7 against the reference; a rank hung for 35 s with an 8 s timeout (the other regroups alone and
finishes on the reference curve; the hung one is ended by the stall watchdog); every job leaves at step 20 and a third
job resumes from the checkpoint on the reference curve; starting from a pool version (equal sha256); the per-GPU lr
rule; a job with a different global batch refused; a hold longer than the collective timeout; EMA updated in
`post_commit` before a regroup.

Not yet measured on GPUs: more than 8 ranks, 2-4 GPU jobs joining, a hang inside a real training step, tuned NCCL
socket settings. Crash recovery takes ~20 s, mostly waiting for the dead job's heartbeat to expire.

Compute: probe 0.11, proof p1 0.46 and p2 ~0.4 GPU-hours; the transport tests of 11.1 0.53 GPU-hours (0.42 of it the
hung tuned run); generator smoke tests 0.98 and the cancelled first launch 1.90 GPU-hours (sacct).

## Acknowledgement

Compute: EuroHPC AI Factory allocation EHPC-AIF-2026PG01-907 on Arrhenius (NAISS, Sweden). The whole benchmark,
including smoke tests, tuning sweeps and failed arms, used 9.33 GPU-hours.
