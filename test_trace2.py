from intellivest.config import load_settings
from intellivest.orchestration.orchestrator import AdvisoryEngine
from intellivest.contracts import InvestorProfile
from datetime import date
s = load_settings()
e = AdvisoryEngine(s, use_ml=True)
p = InvestorProfile(investor_id='T1', as_of=date.fromisoformat(s.data_end), age=34, horizon_years=12, risk_tolerance=4, loss_tolerance_pct=0.2, goal='growth', liquidity_need='low', current_holdings={'SPX':0.35,'TECH':0.3,'GOLD':0.1,'GOVT':0.25})
tr = e.advise(p)
print(tr.ml_forecast is not None)
