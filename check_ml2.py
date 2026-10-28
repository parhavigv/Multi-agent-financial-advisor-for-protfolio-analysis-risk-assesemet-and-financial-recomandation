from intellivest.config import load_settings
from intellivest.data.market import build_provider
from intellivest.quant.engine import QuantEngine
from datetime import date
s = load_settings()
p = build_provider(s)
snap = p.snapshot(date.fromisoformat(s.data_end))
q = QuantEngine(s, use_ml=True)
ml_res = q._ml_service.forecast(snap, s)
q.last_ml_forecast = ml_res
print(q.last_ml_forecast is not None)
