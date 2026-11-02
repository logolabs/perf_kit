# Site environment for bench/slurm/bench_arms.sbatch (copy, fill in, pass as SITE_ENV=/path/to/site_env.sh).
# Everything cluster-specific lives here, nothing in the benchmark scripts. What we used on a GH200 cluster
# (aarch64, CUDA 13, torch 2.13) is described in the comments.

# module load <python module>                       # we used the site's Python 3.13 build
# source <venv>/bin/activate                         # torch 2.13 (cu130), transformers 5.16, peft 0.20, triton 3.7
export PY=python                                     # interpreter for perf_kit / transformers arms
export SWIFT_PY=python                               # interpreter with ms-swift 4.5.3 installed (may be another venv)
# export PERUN=<venv>/bin/perun                      # optional energy monitor; unset = run without it
# export SRUN_FLAGS="--network=single_node_vni"      # extra srun flags your site needs for single-node steps

# fast kernels the perf_kit / HF arms use (pip --target installs work; prepend their dirs):
#   flash-attn 2.8.3 (built for sm_90), flash-linear-attention 0.5.2, liger-kernel 0.8.4, [flash-attn 3 for fa3]
# export PYTHONPATH=<flash-attn dir>:<fla dir>:<liger dir>${PYTHONPATH:+:$PYTHONPATH}

# export TEST_PYTHONPATH=<dir with pytest>          # only for the `tests` arm, if the venv lacks pytest

# offline compute nodes: weights and data staged beforehand
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Triton / TileLang caches on node-local disk
export TRITON_CACHE_DIR=/tmp/$USER/triton XDG_CACHE_HOME=/tmp/$USER/xdg
mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"
# fla's TileLang JIT with a pip-installed nvcc newer than the runtime headers (Hopper):
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
