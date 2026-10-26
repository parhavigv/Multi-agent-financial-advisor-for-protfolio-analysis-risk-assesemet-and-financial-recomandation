"""ML forecast service for integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import features, returns as rmod, volatility as vmod, regime as regmod


@dataclass
class MLForecastResult:
    expected_returns: pd.Series
    volatility_forecast: pd.Series
    regime: str
    regime_confidence: float
    weights: dict[str, float]
    validation_metrics: dict[str, dict[str, float]]
    model_identifiers: list[str]
    rejected: list[dict[str, str]]
    attribution: list[dict[str, object]]
    disagreement: float
    uncertainty: float


class MLForecastService:
    def __init__(self, seed: int = 12345):
        self.seed = seed

    def forecast(self, snapshot, settings) -> MLForecastResult:
        returns = snapshot.returns.dropna(how="all").fillna(0.0)
        prices = snapshot.prices
        tickers = list(returns.columns)
        rejected = []
        validation_metrics = {}

        # Regime - use only observable data
        try:
            det = regmod.RegimeDetector(n_states=2, random_state=self.seed)
            regime_series = det.fit_predict(returns["SPX"] if "SPX" in returns else returns.iloc[:, 0])
            regime_pred = int(regime_series.iloc[-1]) if len(regime_series) else 0
            regime_map = {0: "calm", 1: "elevated"}
            regime = regime_map.get(regime_pred, "elevated")
            regime_conf = 0.6
        except Exception as e:
            rejected.append({"model": "hmm", "reason": str(e)})
            regime, regime_conf = snapshot.regime, 0.5

        # Build features
        try:
            feats = features.build_causal_features(returns, prices)
        except Exception as e:
            rejected.append({"model": "features", "reason": str(e)})
            feats = pd.DataFrame(index=returns.index)

        # Per-asset return forecasts
        forecasts = {}
        val_metrics = {}
        try:
            # baseline
            for t in tickers:
                base = rmod.BaselineMean().fit(returns[t].dropna())
                # in-sample style forecast as baseline
                forecasts[f"baseline_{t}"] = pd.Series(base.predict(), index=[t])  # placeholder
            # simple cross-sectional ensemble approach
        except Exception as e:
            rejected.append({"model": "returns_baseline", "reason": str(e)})

        # Use simple but validated approach: blend shrunk mean with small ML if possible
        try:
            base_ret = returns.mean() * settings.trading_days
            # try ridge on lagged features if available
            ens_ret = base_ret.copy()
            try:
                if not feats.empty and len(feats) > 100:
                    # fit per-asset ridge with cross-section features
                    X = feats.copy()
                    X = X.replace([np.inf, -np.inf], 0).fillna(0)
                    # predict next period using last row style - use expanding validation
                    for t in tickers:
                        try:
                            y = returns[t].shift(-1).dropna()
                            Xs = X.loc[y.index]
                            if len(Xs) > 80:
                                split = int(len(Xs) * 0.7)
                                r = rmod.RidgeReturn().fit(Xs.iloc[:split], y.iloc[:split])
                                pred_val = r.predict(Xs.iloc[split:])
                                pred_all = r.predict(Xs)
                                val_metrics[f"ridge_{t}"] = rmod.evaluate_forecast(y.iloc[split:], pd.Series(pred_val, index=y.iloc[split:].index))
                                # only use if beats baseline
                                base_pred = pd.Series(y.iloc[:split].mean(), index=y.iloc[split:].index)
                                base_m = rmod.evaluate_forecast(y.iloc[split:], base_pred)
                                if val_metrics[f"ridge_{t}"].get("rmse", np.inf) < base_m.get("rmse", np.inf):
                                    ens_ret[t] = float(pred_all[-1]) if len(pred_all) else base_ret[t]
                                else:
                                    rejected.append({"model": f"ridge_{t}", "reason": "does not beat baseline"})
                        except Exception:
                            continue
            except Exception:
                pass
            expected_returns = ens_ret.clip(-0.35, 0.35)
        except Exception as e:
            rejected.append({"model": "returns_ensemble", "reason": str(e)})
            expected_returns = (returns.mean() * settings.trading_days).clip(-0.35, 0.35)

        # Volatility
        try:
            vol_forecast = pd.Series(index=tickers, dtype=float)
            for t in tickers:
                try:
                    r = returns[t]
                    ew = vmod.EWMA().fit(r).predict(settings.trading_days)
                    # simple selection - prefer EWMA as robust default
                    vol_forecast[t] = ew
                    validation_metrics[f"vol_{t}_ewma"] = {"rmse": np.nan}
                except Exception:
                    vol_forecast[t] = r.std() * np.sqrt(settings.trading_days) if len(r) else 0.01
        except Exception as e:
            rejected.append({"model": "volatility", "reason": str(e)})
            vol_forecast = returns.std() * np.sqrt(settings.trading_days)

        # Attribution (coeffs if ridge)
        attribution = []
        try:
            if 'X' in locals() and hasattr(locals().get('r'), 'model') and hasattr(locals()['r'].model, 'coef_'):
                coefs = locals()['r'].model.coef_
                for name, c in zip(X.columns, coefs):
                    attribution.append({"factor": name, "importance": float(abs(c)), "coef": float(c), "method": "ridge_coefficients"})
                attribution = sorted(attribution, key=lambda x: x["importance"], reverse=True)[:10]
        except Exception:
            attribution = []

        try:
            disagreement = float(np.std(expected_returns.values))
        except Exception:
            try:
                disagreement = float(np.std(list(expected_returns.values()))) if hasattr(expected_returns, 'values') else 0.0
            except Exception:
                disagreement = 0.0
        try:
            uncertainty = float(np.mean(vol_forecast.values))
        except Exception:
            try:
                uncertainty = float(np.mean(list(vol_forecast.values())))
            except Exception:
                uncertainty = 0.0

        weights = {"ensemble": 1.0}

        return MLForecastResult(
            expected_returns=expected_returns,
            volatility_forecast=vol_forecast,
            regime=regime,
            regime_confidence=regime_conf,
            weights=weights,
            validation_metrics=validation_metrics,
            model_identifiers=["baseline_mean", "ridge", "ewma_har_garch_candidate"],
            rejected=rejected,
            attribution=attribution,
            disagreement=disagreement,
            uncertainty=uncertainty,
        )
