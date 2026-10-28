"""Forecasting model interface and panel containers.

Every model  -  baseline or candidate  -  implements the same tiny contract:

    model.fit(panel_train)
    preds = model.predict(panel_predict)   # pd.Series indexed by (date, ticker)

Keeping the interface identical is what makes the comparison honest: the
baseline cannot hide behind a different evaluation path, and a candidate can
only enter the production pipeline by beating the baselines out of sample.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PanelIndex = pd.MultiIndex


def make_panel_index(dates: pd.DatetimeIndex, tickers: list[str]) -> PanelIndex:
    """Stack a (date x ticker) block into a MultiIndex usable by sklearn."""
    return pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])


def wide_to_long(frame: pd.DataFrame) -> pd.Series:
    """(dates x tickers) -> Series indexed by (date, ticker), ticker-major."""
    if frame.empty:
        return pd.Series(dtype=float, name="value")
    stacked = frame.stack(future_stack=True)
    stacked.index = stacked.index.set_names(["date", "ticker"])
    return stacked.sort_index()


def long_to_wide(series: pd.Series) -> pd.DataFrame:
    """Series indexed by (date, ticker) -> (dates x tickers)."""
    if series.empty:
        return pd.DataFrame()
    return series.unstack("ticker").sort_index()


@dataclass
class PanelData:
    """A stacked, labelled cross-section over time.

    ``X`` has one row per (date, ticker) and one column per feature. ``y`` is
    aligned to ``X``. Rows with a missing target are dropped by construction in
    :mod:`intellivest.forecasting.data`.
    """

    X: pd.DataFrame
    y: pd.Series
    tickers: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tickers and isinstance(self.X.index, pd.MultiIndex):
            self.tickers = sorted(self.X.index.get_level_values("ticker").unique())

    @property
    def dates(self) -> pd.DatetimeIndex:
        if isinstance(self.X.index, pd.MultiIndex):
            return pd.DatetimeIndex(sorted(self.X.index.get_level_values("date").unique()))
        return pd.DatetimeIndex([])

    def take_dates(self, mask: np.ndarray) -> "PanelData":
        dates = self.dates
        keep = set(dates[mask])
        sel = self.X.index.get_level_values("date").isin(keep)
        return PanelData(X=self.X.loc[sel], y=self.y.loc[sel], tickers=self.tickers)

    def between(self, start, end) -> "PanelData":
        dates = self.X.index.get_level_values("date")
        sel = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        return PanelData(X=self.X.loc[sel], y=self.y.loc[sel], tickers=self.tickers)


class ForecastModel(ABC):
    """Common contract for return and volatility forecasters."""

    name: str = "abstract"
    kind: str = "return"  # "return" or "volatility"
    uses_features: bool = True

    @abstractmethod
    def fit(self, train: PanelData) -> "ForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, predict: PanelData) -> pd.Series:
        """Return predictions aligned to ``predict.X.index``."""
        raise NotImplementedError

    # -- introspection ----------------------------------------------------
    def coefficients(self) -> dict[str, float]:
        """Optional linear coefficients, used for trace attribution."""
        return {}

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "uses_features": self.uses_features}


__all__ = [
    "ForecastModel",
    "PanelData",
    "make_panel_index",
    "wide_to_long",
    "long_to_wide",
]
