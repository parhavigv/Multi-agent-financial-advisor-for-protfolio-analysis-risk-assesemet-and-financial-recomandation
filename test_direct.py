from intellivest.config import load_settings
from intellivest.data.market import build_provider
from datetime import date
from intellivest.quant.engine import QuantEngine
s = load_settings()
p = build_provider(s)
snap = p.snapshot(date.fromisoformat(s.data_end))
q = QuantEngine(s, use_ml=True)
try:
    ml_res = q._ml_service.forecast(snap, s)
    q.last_ml_forecast = ml_res
    print('ok')
except Exception as e:
    import traceback
    traceback.print_exc()
