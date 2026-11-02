"""Print report/index.html to report/perf_kit_report_2026-11.pdf (A4, headless Chromium, mono page footer).

    python bench/analysis/print_report.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from house import print_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    out = ROOT / "report" / "perf_kit_report_2026-11.pdf"
    print_pdf(ROOT / "report" / "index.html", out, "perf_kit technical report · LogoLabs · November 2026")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
