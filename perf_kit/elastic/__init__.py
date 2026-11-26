"""Elastic synchronous data parallelism over plain TCP between SEPARATE Slurm jobs (perf_kit advanced mode).

Many small jobs (1-4 GPUs each) train ONE model with ordinary per-step (or every-k-micro-batches) gradient
all-reduce; jobs join and leave at any time. Each membership epoch is a fresh torch.distributed group (NCCL over its
socket transport between jobs, NVLink inside a job); the membership itself lives in files of a shared run folder.
See core.py (API + protocol), membership.py (files), state.py (state transfer, hashes, checkpoints) and
example.py (worked example / test vehicle). The asynchronous DiLoCo pool (perf_kit.pool) is the other mode.
"""
__all__ = ["ElasticTCP", "is_dist_error"]


def __getattr__(name):                   # lazy: the submit helper runs on login nodes without torch
    if name in __all__:
        from . import core
        return getattr(core, name)
    raise AttributeError(name)
