"""Run the forecast-model evaluation and write ``results/forecast_report.md``.

Development runs (the default) never read the test window. The final report
run passes ``--include-test`` exactly once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intellivest.forecasting.evaluation import evaluate, write_report  # noqa: E402


def load_returns(prices_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not prices_path.exists():
        raise SystemExit(
            f"missing {prices_path}; run `python scripts/fetch_real_data.py` first"
        )
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    if prices.empty:
        raise SystemExit(f"{prices_path} contains no rows")
    returns = prices.pct_change().dropna(how="all")
    return returns, prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prices",
        type=Path,
        default=ROOT / "data" / "prices.csv",
        help="cached price panel (date x ticker)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "forecast_report.md",
        help="Markdown report destination (JSON written alongside)",
    )
    parser.add_argument("--horizon", type=int, default=21, help="forecast horizon in trading days")
    parser.add_argument("--min-train", type=int, default=378, help="minimum training dates")
    parser.add_argument("--embargo", type=int, default=21, help="embargo dates after the purge")
    parser.add_argument("--step", type=int, default=21, help="prediction block size in dates")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-states", type=int, default=3, help="maximum HMM states")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="evaluate the 2024+ test window (must be run exactly once for the final report)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    returns, prices = load_returns(args.prices)
    result = evaluate(
        returns,
        prices,
        horizon=args.horizon,
        min_train=args.min_train,
        embargo=args.embargo,
        step=args.step,
        include_test=args.include_test,
        seed=args.seed,
        max_states=args.max_states,
    )
    write_report(result, args.output)
    print(f"wrote {args.output}")
    print(f"test included: {result.include_test}")
    print(f"selected return model: {result.selected_return}")
    print(f"selected volatility model: {result.selected_vol}")
    print(f"elapsed: {result.elapsed_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
