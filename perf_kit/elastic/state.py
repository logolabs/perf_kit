"""Training state of one ElasticTCP run: what a newcomer receives, how it is hashed and checkpointed.

The state is: model parameters + buffers, any extra modules (EMA, style encoder, ...), the optimizer's state_dict
(moments AND param_groups, so a scheduler's lr written into the groups travels too), the counters (optimizer step,
global samples, global micro-batch cursor) and a picklable `extra` dict (data cursors, scheduler state, anything).

Transfer at an epoch change (`broadcast_state`): the source rank pickles the optimizer's state_dict with every tensor
replaced by a (shape, dtype, device) placeholder and broadcasts that plus counters + extra
(broadcast_object_list); then all tensors go in one flat buffer per dtype (chunks of <= 256 MB), and every rank
copies them into its own tensors and loads the optimizer state. A newcomer's optimizer has no state yet; after
load_state_dict it is identical to the source's. Verification (`fingerprint`): sha256 of parameter + buffer bytes,
compared on every rank against the source's.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path

import torch
import torch.distributed as dist

_CHUNK = 256 * 1024 * 1024


class _T:                                                  # placeholder for a tensor inside a pickled structure
    __slots__ = ("i", "shape", "dtype", "cpu")

    def __init__(self, i, t):
        self.i, self.shape, self.dtype, self.cpu = i, tuple(t.shape), t.dtype, t.device.type == "cpu"


def _strip(obj, out: list):
    if torch.is_tensor(obj):
        out.append(obj)
        return _T(len(out) - 1, obj)
    if isinstance(obj, dict):
        return {k: _strip(v, out) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        r = [_strip(v, out) for v in obj]
        return type(obj)(r) if isinstance(obj, tuple) else r
    return obj


def _fill(obj, ts: list):
    if isinstance(obj, _T):
        return ts[obj.i]
    if isinstance(obj, dict):
        return {k: _fill(v, ts) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        r = [_fill(v, ts) for v in obj]
        return type(obj)(r) if isinstance(obj, tuple) else r
    return obj


def module_tensors(modules: dict) -> list[torch.Tensor]:
    """Parameters then buffers of each module, in a fixed order (sorted module keys, registration order)."""
    out = []
    for k in sorted(modules):
        m = modules[k]
        out += [p.data for p in m.parameters()]
        out += [b for b in m.buffers() if b is not None]
    return out


def _bcast_flat(tensors: list[torch.Tensor], src: int, device: torch.device) -> None:
    """Broadcast tensors from `src` in place: one flat buffer per dtype, chunks of <= 256 MB."""
    groups = {}
    for t in tensors:
        groups.setdefault(t.dtype, []).append(t)
    for dt, ts in groups.items():
        i = 0
        while i < len(ts):
            j, nbytes = i, 0
            while j < len(ts) and (j == i or nbytes + ts[j].numel() * ts[j].element_size() <= _CHUNK):
                nbytes += ts[j].numel() * ts[j].element_size()
                j += 1
            part = ts[i:j]
            flat = torch.cat([t.detach().reshape(-1).to(device) for t in part]) if dist.get_rank() == src else \
                torch.empty(sum(t.numel() for t in part), dtype=dt, device=device)
            dist.broadcast(flat, src=src)
            o = 0
            with torch.no_grad():
                for t in part:
                    n = t.numel()
                    t.copy_(flat[o:o + n].view_as(t))
                    o += n
            del flat
            i = j


def broadcast_state(modules: dict, optimizer, counters: dict, extra: dict, src: int, device: torch.device) -> dict:
    """Make every rank's state equal to rank `src`'s. Returns {counters, extra} as received (rank src: its own)."""
    rank = dist.get_rank()
    opt_ts: list[torch.Tensor] = []
    meta = None
    if rank == src:
        sd = optimizer.state_dict() if optimizer is not None else None
        sd_s = _strip(sd, opt_ts)
        meta = {"opt": sd_s, "counters": counters, "extra": extra}
    box = [pickle.dumps(meta) if rank == src else None]
    dist.broadcast_object_list(box, src=src, device=device if device.type == "cuda" else None)
    meta = pickle.loads(box[0])
    if rank != src:                                        # allocate the optimizer tensors to receive into
        opt_ts = [None] * _count(meta["opt"])
        _alloc(meta["opt"], opt_ts)
    _bcast_flat(module_tensors(modules), src, device)
    if opt_ts:
        _bcast_flat(opt_ts, src, device)
    if optimizer is not None and rank != src and meta["opt"] is not None:
        optimizer.load_state_dict(_fill(meta["opt"], opt_ts))
    return {"counters": meta["counters"], "extra": meta["extra"]}


def _count(obj) -> int:
    n = [0]

    def walk(o):
        if isinstance(o, _T):
            n[0] = max(n[0], o.i + 1)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return n[0]


def _alloc(obj, out: list, device=None):
    if isinstance(obj, _T):
        dev = "cpu" if obj.cpu else (device or (torch.device("cuda", torch.cuda.current_device())
                                                if torch.cuda.is_available() else "cpu"))
        out[obj.i] = torch.empty(obj.shape, dtype=obj.dtype, device=dev)
    elif isinstance(obj, dict):
        for v in obj.values():
            _alloc(v, out, device)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _alloc(v, out, device)


def fingerprint(modules: dict) -> str:
    """sha256 over the bytes of every parameter and buffer (CPU copies): equal state, equal hash, on any rank."""
    h = hashlib.sha256()
    for t in module_tensors(modules):
        c = t.detach().contiguous().cpu()
        h.update(str((tuple(c.shape), str(c.dtype))).encode())
        h.update(c.reshape(-1).view(torch.uint8).numpy().tobytes() if c.numel() else b"")
    return h.hexdigest()


def quick_checksum(modules: dict) -> float:
    """float64 sum of all parameters (+ an index weight per tensor): cheap, deterministic, for the per-step desync
    check (bit-identical replicas give bit-identical sums)."""
    s = None
    for i, t in enumerate(module_tensors(modules)):
        if not torch.is_floating_point(t):
            continue
        v = t.detach().double().sum() * (1.0 + 1e-3 * (i % 997))
        s = v if s is None else s + v
    return s if s is not None else torch.zeros((), dtype=torch.float64)


# ---------------------------------------------------------------- checkpoints
def cpu_state(modules: dict, optimizer, counters: dict, extra: dict) -> dict:
    def cpu(o):
        if torch.is_tensor(o):
            return o.detach().to("cpu", copy=True)
        if isinstance(o, dict):
            return {k: cpu(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(cpu(v) for v in o)
        return o
    return {"modules": {k: cpu(m.state_dict()) for k, m in modules.items()},
            "optimizer": cpu(optimizer.state_dict()) if optimizer is not None else None,
            "counters": dict(counters), "extra": cpu(extra), "t": time.time()}


def save_checkpoint(ckpt_dir: Path, state: dict, keep: int = 2) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"step_{int(state['counters']['step']):09d}.pt"
    tmp = final.with_name(f".{final.name}.{os.getpid()}.tmp")
    torch.save(state, tmp)
    os.replace(tmp, final)
    olds = sorted(ckpt_dir.glob("step_*.pt"))
    for p in olds[:-keep] if keep > 0 else []:
        try:
            os.unlink(p)
        except OSError:
            pass
    return final


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    c = sorted(Path(ckpt_dir).glob("step_*.pt"))
    return c[-1] if c else None


def load_checkpoint(path: Path, modules: dict, optimizer) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    for k, m in modules.items():
        m.load_state_dict(d["modules"][k])
    if optimizer is not None and d.get("optimizer") is not None:
        optimizer.load_state_dict(d["optimizer"])
    return d
