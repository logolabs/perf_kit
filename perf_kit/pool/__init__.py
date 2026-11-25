"""A growable GPU pool for one training run on a busy Slurm cluster (perf_kit's advanced mode).

Intentionally designed to maximise the use of GPUs that are only partly used: the free GPUs on partly allocated
nodes and the short free windows of a busy cluster. Not for short runs (few outer rounds measured clearly worse
than DDP) or when whole nodes start quickly; see README "Advanced mode" and WRITEUP section 10.

Many small jobs (1..N GPUs each, ordinary DDP inside) that start whenever GPUs are free work on ONE run; they join
and leave at any time, and a big multi-node job can join the same run when it starts. Between jobs there is no
NCCL: each job trains `inner_steps` local steps from the current global weights, drops its pseudo-gradient
(global - local) into the run folder on the shared filesystem, and whoever syncs next folds the inbox into a new
global version with an outer Nesterov optimizer (DiLoCo, Douillard et al. 2023, arXiv 2311.08105; asynchronous
Delayed Nesterov, Liu et al. 2024, arXiv 2401.09135). See outer.py for the math, store.py for the files, worker.py
for the API and launch.py for keeping the pool topped up under a GPU cap.
"""
__all__ = ["Pool", "Store", "LockBusy", "OuterConfig", "OuterState", "Contribution", "apply"]
_WHERE = {"Pool": "worker", "Store": "store", "LockBusy": "store", "OuterConfig": "outer", "OuterState": "outer",
          "Contribution": "outer", "apply": "outer"}


def __getattr__(name):                   # lazy, so `python -m perf_kit.pool.launch` needs no torch (login nodes)
    if name in _WHERE:
        import importlib
        return getattr(importlib.import_module(f".{_WHERE[name]}", __name__), name)
    raise AttributeError(name)
