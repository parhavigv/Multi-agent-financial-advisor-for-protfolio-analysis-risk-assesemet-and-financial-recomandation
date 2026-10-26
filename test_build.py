from intellivest.config import load_settings
from intellivest.data.market import build_provider
from intellivest.quant.engine import QuantEngine
from datetime import date
s = load_settings()
p = build_provider(s)
snap = p.snapshot(date.fromisoformat(s.data_end))
q = QuantEngine(s, use_ml=True)
f = q.build(snap)
print(q.last_ml_forecast)
