#!/usr/bin/env python
"""Reproducible experiment entry point.

Thin wrapper around ``quantumrisklab.experiments.runner:main`` so the sweep can be
run without installing the package:

    python scripts/run_experiments.py --sizes 10 20 30 50 75

Configuration (database URL, seed, generation params) comes from environment
variables / ``.env`` via ``quantumrisklab.config``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without `pip install`.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantumrisklab.experiments.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
