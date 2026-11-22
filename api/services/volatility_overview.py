"""Simple, dated Cboe index observations sourced through Yahoo Finance."""
from datetime import datetime, timezone
import math
import yfinance as yf
from api.services.options_flow_calendar import NY, previous_trading_day, is_trading_day

def overview_from_series(series, now=None):
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(NY)
    day = local.date()
    if not is_trading_day(day) or (local.hour, local.minute) < (16, 30):
        day = previous_trading_day(day)
    def clean(rows):
        out = {}
        for stamp, value in rows:
            try:
                value = float(value)
                key = str(stamp)[:10]
                if math.isfinite(value) and value > 0 and key <= day.isoformat():
                    out[key] = value
            except (ValueError, TypeError):
                pass
        return out
    vix, three = clean(series.get('vix', [])), clean(series.get('vix3m', []))
    points = [{'date':d,'vix':round(vix[d],2),'vix3m':round(three[d],2) if d in three else None,
               'ratio':round(vix[d]/three[d],4) if d in three else None} for d in sorted(vix)][-252:]
    if not points:
        raise ValueError('No completed VIX observations; previous snapshot retained')
    latest = points[-1]
    return {'as_of':latest['date'],'updated_at':now.isoformat(),'source':'Yahoo Finance / yfinance',
            'symbols':{'vix':'^VIX','vix3m':'^VIX3M'},'points':points,
            'latest':dict(latest,change_points=round(latest['vix']-points[-2]['vix'],2) if len(points)>1 else None),
            'expected_session':day.isoformat(),'stale':latest['date']<day.isoformat()}

def build_overview():
    series = {}
    for key,ticker in [('vix','^VIX'),('vix3m','^VIX3M')]:
        frame = yf.Ticker(ticker).history(period='2y',interval='1d',auto_adjust=False,actions=False,timeout=25)
        series[key] = [(str(d.date()),v) for d,v in frame['Close'].items()] if not frame.empty else []
    return overview_from_series(series)
