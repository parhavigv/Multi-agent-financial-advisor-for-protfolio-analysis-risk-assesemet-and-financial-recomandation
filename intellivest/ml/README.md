# ML Package (IntelliVest)

This package provides causal, leakage-safe ML forecasting components:

- `base.py`: metrics (RMSE/MAE/QLIKE/IC/RankIC/DM/block bootstrap)
- `features.py`: causal feature engineering with lagging
- `splits.py`: chronological splits with purge/embargo
- `returns.py`: baseline/ridge/GBM with ensemble fallback
- `volatility.py`: EWMA/HAR/GARCH(1,1) with validation
- `regime.py`: HMM (with BIC) using observable data only
- `ensemble.py`: inverse-error weighting + shrinkage
- `walkforward.py`: expanding-window utilities
- `service.py`: integration service used by QuantEngine

## Leakage prevention
- Features are shifted/lagged; no future values used
- Models fit on training data only
- Regime detection uses only observed history (no hidden simulator state)
- Truncation invariance: earlier timestamps unchanged by future data
