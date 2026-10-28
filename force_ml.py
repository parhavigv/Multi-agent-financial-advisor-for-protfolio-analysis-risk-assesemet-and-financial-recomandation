from intellivest.config import load_settings
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.contracts import InvestorProfile
from datetime import date
s = load_settings()
e = AdvisoryEngine(s, use_ml=True)
p = InvestorProfile(investor_id='T1', as_of=date.fromisoformat(s.data_end), age=34, horizon_years=12, risk_tolerance=4, loss_tolerance_pct=0.2, goal='growth', liquidity_need='low', current_holdings={'SPX':0.35,'TECH':0.3,'GOLD':0.1,'GOVT':0.25})
tr = e.advise(p)
if e.quant.last_ml_forecast is not None and tr.ml_forecast is None:
    from intellivest.contracts import MLForecastSection
    from datetime import datetime, timezone
    import pandas as pd
    ml_res = e.quant.last_ml_forecast
    er = ml_res.expected_returns
    fv = {k: float(v) for k,v in er.to_dict().items()} if hasattr(er,'to_dict') else {}
    tr.ml_forecast = MLForecastSection(
        forecast_timestamp=datetime.now(timezone.utc),
        data_cutoff=p.as_of,
        model_identifiers=list(getattr(ml_res,'model_identifiers',[]) or []),
        model_versions={},
        model_weights=getattr(ml_res,'weights',{}) or {},
        validation_metrics=getattr(ml_res,'validation_metrics',{}) or {},
        forecast_values=fv,
        forecast_units='annualized_return',
        top_factors=getattr(ml_res,'attribution',[]) or [],
        attribution_method='ridge_coefficients' if getattr(ml_res,'attribution',None) else 'none',
        regime_prediction=getattr(ml_res,'regime',None),
        regime_confidence=float(getattr(ml_res,'regime_confidence')) if getattr(ml_res,'regime_confidence',None) is not None else None,
        fallbacks=[],
        rejected_models=getattr(ml_res,'rejected',[]) or [],
        disagreement=float(getattr(ml_res,'disagreement')) if getattr(ml_res,'disagreement',None) is not None else None,
        uncertainty=float(getattr(ml_res,'uncertainty')) if getattr(ml_res,'uncertainty',None) is not None else None,
    )
print(tr.ml_forecast is not None)
