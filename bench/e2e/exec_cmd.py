"""Run a command as a Python script, for wrappers that only accept a script (e.g. `perun monitor ... exec_cmd.py
swift sft ...`): returns normally on success (some energy monitors treat any SystemExit, even 0, as a failure and
drop the record) and raises on failure."""
from __future__ import annotations

import subprocess
import sys


def main(argv: list[str]) -> int:
    if not argv:
        raise RuntimeError("usage: exec_cmd.py <command> [args...]")
    return subprocess.run(argv).returncode


if __name__ == "__main__":
    rc = main(sys.argv[1:])
    if rc:
        raise RuntimeError(f"command failed with exit code {rc}: {' '.join(sys.argv[1:])[:400]}")
