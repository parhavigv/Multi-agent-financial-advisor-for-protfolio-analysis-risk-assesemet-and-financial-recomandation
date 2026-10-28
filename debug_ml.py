from intellivest.config import load_settings
from intellivest.data.market import build_provider
from datetime import date
from intellivest.ml import service
s = load_settings()
p = build_provider(s)
snap = p.snapshot(date.fromisoformat(s.data_end))
svc = service.MLForecastService(s.seed)
try:
    res = svc.forecast(snap, s)
    print('ok')
except Exception as e:
    import traceback
    traceback.print_exc()
