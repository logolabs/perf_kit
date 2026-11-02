"""Script entry for tools that need a file path rather than `python -m` (e.g. `perun monitor run_train.py ...`).
Ends with `rc = main(); if rc: sys.exit(rc)` so energy monitors that treat any SystemExit as failure still record."""
import os
import sys

sys.path.insert(0, os.environ.get("PERF_KIT", os.path.dirname(os.path.abspath(sys.argv[0]))))
from perf_kit.train import main  # noqa: E402

rc = main()
if rc:
    sys.exit(rc)
