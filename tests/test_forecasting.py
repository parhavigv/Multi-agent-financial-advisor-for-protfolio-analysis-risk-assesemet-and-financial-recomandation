"""Leakage, causality and interface tests for the forecasting package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from intellivest.forecasting import metrics as M
from intellivest.forecasting.base import ForecastModel, PanelData
from intellivest.forecasting.cv import out_of_fold_predictions, purged_expanding_folds
from intellivest.forecasting.data import build_features, build_panel, forward_return
from intellivest.forecasting.evaluation import _last_fold_info, _split_dates, _train_metrics
from intellivest.forecasting.regime import HMMRegime
from intellivest.forecasting.return_models import (
    BASELINES,
    CANDIDATES,
    MomentumForecast,
    OneOverNForecast,
    RidgeForecast,
    ShrunkMeanForecast,
)
from intellivest.forecasting.vol_models import EWMAVolatility, GARCH11Volatility, HARVolatility


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2018-01-01", periods=n)


def _noise_returns(n: int = 700, tickers: int = 12, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _dates(n)
    names = [f"A{i:02d}" for i in range(tickers)]
    common = rng.normal(0.0, 0.006, size=(n, 1))
    returns = common + rng.normal(0.0, 0.012, size=(n, tickers))
    return pd.DataFrame(returns, index=dates, columns=names)


def _momentum_returns(n: int = 900, tickers: int = 12, seed: int = 7) -> pd.DataFrame:
    """Returns with a planted cross-sectional momentum signal.

    ``mu`` at date t is proportional to each asset's own trailing 21-day return
    (lagged), so a model that uses momentum can predict the cross-section. The
    data are synthetic; the point is to test the *pipeline*, not the market.
    """
    rng = np.random.default_rng(seed)
    dates = _dates(n)
    names = [f"A{i:02d}" for i in range(tickers)]
    common = rng.normal(0.0, 0.004, size=(n, 1))
    returns = np.zeros((n, tickers))
    for t in range(n):
        if t > 25:
            past = returns[max(0, t - 21) : t].sum(axis=0)
            signal = (past - past.mean()) / (past.std() + 1e-9)
            mu = 0.004 * signal
        else:
            mu = np.zeros(tickers)
        returns[t] = mu + common[t, 0] + rng.normal(0.0, 0.010, size=tickers)
    return pd.DataFrame(returns, index=dates, columns=names)


def _panel(returns: pd.DataFrame, kind: str = "return", horizon: int = 5) -> PanelData:
    prices = (1.0 + returns).cumprod() * 100.0
    return build_panel(returns, prices, kind=kind, horizon=horizon).to_panel()


def _split_panel(panel: PanelData, fraction: float = 0.6) -> tuple[PanelData, PanelData]:
    dates = panel.dates
    assert len(dates) > 120
    cut = max(60, int(len(dates) * fraction))
    train = panel.take_dates(np.arange(len(dates)) < cut)
    test = panel.take_dates(np.arange(len(dates)) >= cut)
    return train, test


# ---------------------------------------------------------------------------
# causality / leakage
# ---------------------------------------------------------------------------


def test_features_do_not_change_when_future_data_is_appended() -> None:
    returns = _noise_returns(n=420)
    horizon = 5
    full = build_features(returns, (1.0 + returns).cumprod())
    truncated = build_features(returns.iloc[:360], (1.0 + returns.iloc[:360]).cumprod())
    common = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[common], truncated, check_exact=False, atol=1e-12
    )


def test_forward_target_only_uses_the_future_window() -> None:
    returns = _noise_returns(n=120, tickers=4)
    horizon = 5
    target = forward_return(returns, horizon)
    t = returns.index[50]
    expected = float(np.prod(1.0 + returns.iloc[51 : 51 + horizon, 0]) - 1.0)
    assert target.loc[t, returns.columns[0]] == pytest.approx(expected, abs=1e-12)
    # The final `horizon` rows have no complete forward window.
    assert target.iloc[-horizon:].isna().all().all()


def test_purged_folds_never_let_train_targets_overlap_prediction() -> None:
    dates = _dates(500)
    horizon, embargo = 5, 5
    folds = purged_expanding_folds(
        dates, min_train=120, horizon=horizon, embargo=embargo, step=10
    )
    assert folds
    for fold in folds:
        gap = (fold.predict_dates.min() - fold.train_dates.max()).days
        # Calendar days must cover at least the purge plus embargo in trading days.
        assert gap >= (horizon + embargo) * 0.9
        assert fold.train_dates.max() < fold.predict_dates.min()


def test_out_of_fold_predictions_have_unique_dates() -> None:
    returns = _noise_returns(n=900)
    panel = _panel(returns, horizon=5)
    predictions, truth, info = out_of_fold_predictions(
        panel,
        lambda: ShrunkMeanForecast(),
        min_train=200,
        horizon=5,
        embargo=5,
        step=20,
    )
    assert info
    assert predictions.index.equals(truth.index)
    assert predictions.index.is_unique
    dates = predictions.index.get_level_values("date")
    assert dates.nunique() == len(dates) // len(panel.tickers)
    assert predictions.notna().all()


def test_max_date_excludes_the_test_window_from_folds() -> None:
    returns = _momentum_returns(n=1400)
    panel = _panel(returns, horizon=5)
    cutoff = pd.Timestamp("2022-12-31")
    predictions, _, _ = out_of_fold_predictions(
        panel,
        lambda: ShrunkMeanForecast(),
        min_train=200,
        horizon=5,
        embargo=5,
        step=20,
        max_date=cutoff,
    )
    assert predictions.index.get_level_values("date").max() <= cutoff


def test_selection_diagnostics_are_invariant_to_including_test() -> None:
    """Adding the test window must not change the folds used for flags.

    A fold that starts before the validation boundary but predicts into the test
    window must be excluded from the selection diagnostics; otherwise the
    in-sample train metrics (and therefore the flags) change on a final run.
    The board always runs on the full causal panel; a development run only
    skips the folds whose prediction block ends after ``max_date``.
    """
    horizon = 5
    returns = _momentum_returns(n=1400)
    panel = _panel(returns, horizon=horizon)
    dates = panel.dates
    val_end = dates[int(len(dates) * 0.7)]

    def run(max_date):
        return out_of_fold_predictions(
            panel,
            lambda: RidgeForecast(horizon=horizon),
            min_train=200,
            horizon=horizon,
            embargo=horizon,
            step=20,
            max_date=max_date,
        )

    _, _, dev_info = run(val_end)
    _, _, full_info = run(None)

    dev_last = _last_fold_info(dev_info, max_date=val_end)
    full_last = _last_fold_info(full_info, max_date=val_end)
    assert dev_last is not None and full_last is not None
    assert dev_last["cutoff"] == full_last["cutoff"]
    assert dev_last["predict_end"] == full_last["predict_end"]
    pd.testing.assert_series_equal(
        dev_last["train_pred"].sort_index(), full_last["train_pred"].sort_index()
    )
    assert _train_metrics(dev_info, "return", horizon, max_date=val_end) == pytest.approx(
        _train_metrics(full_info, "return", horizon, max_date=val_end)
    )


def test_split_masks_exclude_targets_that_cross_the_boundary() -> None:
    dates = _dates(1700)
    horizon = 21
    splits = _split_dates(dates, horizon)
    dev_end = pd.Timestamp("2021-12-31")
    val_end = pd.Timestamp("2023-12-31")
    assert len(splits["dev"]) and len(splits["validation"]) and len(splits["test"])
    assert splits["dev"].max() <= dev_end
    assert splits["validation"].max() <= val_end
    # The final 21 feature dates before the boundary are excluded, so no scored
    # target reaches into the next split.
    dev_cut = dates[dates <= dev_end][-horizon - 1]
    val_cut = dates[dates <= val_end][-horizon - 1]
    assert splits["dev"].max() <= dev_cut
    assert splits["validation"].max() <= val_cut
    assert splits["test"].min() >= pd.Timestamp("2024-01-01")


# ---------------------------------------------------------------------------
# shuffled control
# ---------------------------------------------------------------------------


def test_shuffled_labels_destroy_cross_sectional_signal() -> None:
    # More assets and a longer sample shrink the variance of the random
    # coefficient direction; a 12-asset, 150-date control is too noisy to be a
    # meaningful test.
    returns = _momentum_returns(n=1600, tickers=24)
    panel = _panel(returns, horizon=5)
    kwargs = dict(min_train=300, horizon=5, embargo=5, step=40)

    real_pred, real_truth, _ = out_of_fold_predictions(
        panel, lambda: RidgeForecast(horizon=5), **kwargs
    )
    real_ic = M.return_metrics(real_truth, real_pred, horizon=5)["rank_ic"]
    assert np.isfinite(real_ic) and real_ic > 0.10

    control_ics = []
    for seed in (12345, 1, 2):
        shuffled_pred, shuffled_truth, _ = out_of_fold_predictions(
            panel,
            lambda: RidgeForecast(horizon=5),
            shuffle_train_labels=True,
            seed=seed,
            **kwargs,
        )
        m = M.return_metrics(shuffled_truth, shuffled_pred, horizon=5)
        control_ics.append(m["rank_ic"])
    mean_ic = float(np.mean(control_ics))
    assert abs(mean_ic) < 0.08, f"control mean IC {mean_ic:.4f} is too large"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_metrics_are_zero_for_a_perfect_forecast() -> None:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=40)
    tickers = [f"T{i}" for i in range(6)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    truth = pd.Series(rng.normal(0.01, 0.04, len(index)), index=index)
    out = M.return_metrics(truth, truth.copy(), horizon=5)
    assert out["rmse"] == pytest.approx(0.0)
    assert out["mae"] == pytest.approx(0.0)
    assert out["rank_ic"] == pytest.approx(1.0)
    assert out["hit_rate"] == pytest.approx(1.0)
    dm = M.diebold_mariano_nw(truth**2, truth**2, lags=4)
    assert dm["dm_stat"] == pytest.approx(0.0)


def test_constant_forecast_has_zero_rank_ic() -> None:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2021-01-01", periods=20)
    tickers = [f"T{i}" for i in range(8)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    truth = pd.Series(rng.normal(0.01, 0.04, len(index)), index=index)
    constant = pd.Series(0.01, index=index)
    out = M.return_metrics(truth, constant, horizon=5)
    assert out["rank_ic"] == 0.0
    assert out["constant"] == 1.0


def test_qlike_prefers_the_true_volatility() -> None:
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2021-01-01", periods=30)
    tickers = [f"T{i}" for i in range(5)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    truth = pd.Series(np.abs(rng.normal(0.18, 0.04, len(index))), index=index)
    good = M.qlike_variance(truth, truth.copy())
    bad = M.qlike_variance(truth, truth * 2.0)
    assert good < bad


def test_rank_ic_tstat_is_finite_and_directionally_correct() -> None:
    values = pd.Series([0.1, 0.15, 0.05, 0.2, 0.12, 0.08, 0.11, 0.09])
    mean, tstat, n = M.mean_tstat_nw(values, lags=2)
    assert n == len(values)
    assert mean == pytest.approx(values.mean())
    assert tstat > 2.0


# ---------------------------------------------------------------------------
# models / regime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        *[cls() for cls in (ShrunkMeanForecast, MomentumForecast, OneOverNForecast)],
        RidgeForecast(horizon=5),
        EWMAVolatility(),
        HARVolatility(horizon=5),
        GARCH11Volatility(horizon=5),
    ],
)
def test_models_conform_to_the_interface(factory) -> None:
    assert isinstance(factory, ForecastModel)
    returns = _noise_returns(n=900)
    panel = _panel(returns, kind=factory.kind, horizon=5)
    train, test = _split_panel(panel)
    model = factory.fit(train)
    preds = model.predict(test)
    assert preds.index.equals(test.X.index)
    assert np.isfinite(preds.to_numpy()).all()


def test_hmm_filtered_probabilities_are_prefix_consistent() -> None:
    returns = _noise_returns(n=600, tickers=4)
    benchmark = returns.mean(axis=1)
    train = benchmark.iloc[:300]
    detector = HMMRegime(max_states=3, random_state=12345).fit(train)
    full = detector.filtered_probabilities(benchmark)
    prefix = detector.filtered_probabilities(benchmark.iloc[:350])
    pd.testing.assert_frame_equal(full.iloc[:350], prefix, atol=1e-12)
    # Filtering must not use the future: a truncated input leaves earlier
    # probabilities untouched.
    shorter = detector.filtered_probabilities(benchmark.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], shorter, atol=1e-12)
    labels = detector.labels(benchmark)
    assert set(labels.unique()).issubset({"calm", "elevated", "stress"})


def test_lightgbm_is_available() -> None:
    pytest.importorskip("lightgbm")
    from intellivest.forecasting.return_models import LightGBMForecast

    assert any(cls.name == "lightgbm" for cls in CANDIDATES)
    returns = _noise_returns(n=900)
    panel = _panel(returns, horizon=5)
    train, test = _split_panel(panel)
    model = LightGBMForecast(horizon=5, n_estimators=20).fit(train)
    preds = model.predict(test)
    assert len(preds) > 0
    assert np.isfinite(preds.to_numpy()).all()
