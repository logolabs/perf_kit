"""Build the public benchmark set from HuggingFaceH4/ultrachat_200k (MIT licence).

Each row keeps only the FIRST user -> assistant exchange of a conversation, so every framework trains on exactly one
completion per row (perf_kit trains the last message; ms-swift / TRL train every assistant turn by default -- with
single-turn rows the two coincide). Rows are taken in file order (no shuffling, no filtering beyond role checks and
non-empty text), so the set is fully determined by the dataset revision.

    hf download HuggingFaceH4/ultrachat_200k --repo-type dataset --revision <rev> \
        --include "data/train_sft-00000-of-*.parquet" "data/test_sft-00000-of-*.parquet" --local-dir <src>
    python prepare_ultrachat.py --src <src> --out <dir> [--train_rows 20000] [--val_rows 512] [--revision <rev>]

Writes <out>/train.jsonl, <out>/val.jsonl ({"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}) and
<out>/manifest.json (source, revision, licence, input / output sha256, row counts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_turn(messages) -> list | None:
    if len(messages) < 2:
        return None
    u, a = messages[0], messages[1]
    if u.get("role") != "user" or a.get("role") != "assistant":
        return None
    if not str(u.get("content", "")).strip() or not str(a.get("content", "")).strip():
        return None
    return [{"role": "user", "content": u["content"]}, {"role": "assistant", "content": a["content"]}]


def take(files: list[Path], n: int) -> list:
    import pyarrow.parquet as pq
    out = []
    for f in files:
        tab = pq.read_table(f, columns=["messages"])
        for msgs in tab.column("messages").to_pylist():
            row = first_turn(msgs)
            if row is not None:
                out.append({"messages": row})
                if len(out) >= n:
                    return out
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local dir of the downloaded dataset (contains data/*.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_rows", type=int, default=20000)
    ap.add_argument("--val_rows", type=int, default=512)
    ap.add_argument("--revision", default="", help="dataset commit hash, recorded in the manifest")
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_files = sorted((src / "data").glob("train_sft-*.parquet"))
    va_files = sorted((src / "data").glob("test_sft-*.parquet"))
    if not tr_files or not va_files:
        print(f"no train_sft / test_sft parquet under {src / 'data'}", file=sys.stderr)
        return 2
    manifest = {"dataset": "HuggingFaceH4/ultrachat_200k", "licence": "MIT", "revision": a.revision or None,
                "selection": "first user->assistant exchange of each conversation, file order, no shuffling",
                "inputs": {str(f.relative_to(src)): sha256(f) for f in tr_files + va_files}, "outputs": {}}
    for name, files, n in (("train", tr_files, a.train_rows), ("val", va_files, a.val_rows)):
        rows = take(files, n)
        p = out / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest["outputs"][p.name] = {"rows": len(rows), "sha256": sha256(p)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest["outputs"]))
    return 0


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
