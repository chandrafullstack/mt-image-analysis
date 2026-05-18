"""Smoke test: pricing table is in sync with anything we actually call.

A missing entry would cause `claude_score_crops.score_crops` to raise SystemExit
right at the start of a paid run -- catching it in CI is much cheaper.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.claude_score_crops import PRICES


def test_prices_table_well_formed() -> None:
    for model, price in PRICES.items():
        assert isinstance(model, str) and model.startswith("claude-"), model
        assert isinstance(price, tuple) and len(price) == 2, model
        in_p, out_p = price
        assert 0 < in_p < 1000, f"{model} input price suspicious: {in_p}"
        assert 0 < out_p < 1000, f"{model} output price suspicious: {out_p}"
        assert out_p >= in_p, f"{model} output should not be cheaper than input"


def test_default_model_in_prices() -> None:
    from src.claude_score_crops import _parse_args  # noqa: F401
    # Default model in the parser must be in PRICES.
    # We can't easily fetch defaults without invoking argparse, so re-encode the
    # known default here; if you change the default, change it in both places.
    DEFAULT = "claude-sonnet-4-5-20250929"
    assert DEFAULT in PRICES, f"Default model {DEFAULT} not in PRICES"


if __name__ == "__main__":
    test_prices_table_well_formed()
    test_default_model_in_prices()
    print("OK")
