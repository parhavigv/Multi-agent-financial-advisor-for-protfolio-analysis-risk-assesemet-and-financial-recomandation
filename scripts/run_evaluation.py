"""Run the walk-forward comparison and write results/evaluation.{json,md}."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intellivest.cli import evaluate_main

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
