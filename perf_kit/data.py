"""Generic SFT data for perf_kit.train: chat JSONL rows -> token ids with labels on the completion only.

Accepted rows: {"messages": [{"role": ..., "content": ...}, ...]} (the last message is the target, everything before
it the prompt, rendered with the tokenizer's chat template and its generation prompt), or {"prompt": str,
"completion": str} (plain concatenation). The completion is followed by the tokenizer's end-of-turn token. Rows longer
than max_len are dropped, never truncated.

Encoded rows are kept as one int32 array + offsets + prompt lengths (`Rows`), cached as .npz under a key of the
input files (path, size, mtime), tokenizer and max_len; a cache miss is encoded by a process pool.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

IGNORE = -100


def end_token(tok) -> str:
    for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>"):
        try:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
                return t
        except Exception:                               # noqa: BLE001
            pass
    return tok.eos_token


def encode_row(tok, row: dict, max_len: int, end: str):
    if "messages" in row:
        msgs = row["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        completion = msgs[-1]["content"]
    else:
        prompt, completion = row["prompt"], row["completion"]
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"] + tok(end, add_special_tokens=False)["input_ids"]
    if len(p) + len(c) > max_len:
        return None
    return p + c, len(p)


class Rows:
    def __init__(self, ids, off, plen):
        self.ids, self.off, self.p = ids, off, plen

    def __len__(self):
        return len(self.p)

    def length(self, i):
        return int(self.off[i + 1] - self.off[i])

    def n_labels(self, i):
        return self.length(i) - int(self.p[i])

    def __getitem__(self, i):
        a = self.ids[self.off[i]: self.off[i + 1]].astype(np.int64)
        lab = a.copy()
        lab[: int(self.p[i])] = IGNORE
        seg = np.full(len(a), 2, np.int64)
        seg[: int(self.p[i])] = 0
        return a, lab, seg

    @classmethod
    def build(cls, encs):
        lens = [len(e[0]) for e in encs]
        off = np.zeros(len(encs) + 1, np.int64)
        off[1:] = np.cumsum(lens)
        ids = np.fromiter((x for e in encs for x in e[0]), np.int32, count=int(off[-1]))
        return cls(ids, off, np.array([e[1] for e in encs], np.int32))

    def save(self, path: Path):
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, ids=self.ids, off=self.off, p=self.p)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path):
        z = np.load(path)
        return cls(z["ids"], z["off"], z["p"])


def read_jsonl(paths, limit: int = 0) -> list:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        return rows
    return rows


_W: dict = {}


def _init(tok_src, max_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_src)
    _W.update(tok=tok, max_len=max_len, end=end_token(tok))


def _chunk(rows):
    return [encode_row(_W["tok"], r, _W["max_len"], _W["end"]) for r in rows]


def cache_key(paths, tok, max_len: int, limit: int) -> str:
    files = [(os.path.abspath(p), os.path.getsize(p), int(os.path.getmtime(p))) for p in paths]
    tid = [getattr(tok, "name_or_path", ""), len(tok), str(getattr(tok, "chat_template", ""))[:4000]]
    return hashlib.sha256(json.dumps({"f": files, "t": tid, "m": max_len, "l": limit, "v": 1}).encode()).hexdigest()[:16]


def load(paths, tok, tok_src: str, max_len: int, limit: int = 0, cache_dir: str = "", procs: int = 0):
    """(Rows, dropped, source). procs > 1 encodes a cache miss with a spawned process pool (tok_src reloadable)."""
    key = cache_key(paths, tok, max_len, limit)
    d = Path(cache_dir) / key if cache_dir else None
    if d is not None and (d / "rows.npz").exists():
        return Rows.load(d / "rows.npz"), json.loads((d / "meta.json").read_text())["dropped"], f"cache {key}"
    rows = read_jsonl(paths, limit)
    if procs > 1:
        parts = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
        with multiprocessing.get_context("spawn").Pool(procs, initializer=_init, initargs=(tok_src, max_len)) as pool:
            encs = [e for part in pool.imap(_chunk, parts) for e in part]
    else:
        end = end_token(tok)
        encs = [encode_row(tok, r, max_len, end) for r in rows]
    kept = [e for e in encs if e is not None]
    out = Rows.build(kept)
    if d is not None:
        d.mkdir(parents=True, exist_ok=True)
        out.save(d / "rows.npz")
        (d / "meta.json").write_text(json.dumps({"rows": len(rows), "dropped": len(rows) - len(kept)}))
    return out, len(rows) - len(kept), "encoded"
