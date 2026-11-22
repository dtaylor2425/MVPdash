"""Published observations only. Never reconstruct historical model scores."""
import math
import pandas as pd

def compare_published(rows, prices):
    spy = prices.get('SPY', pd.Series(dtype=float)).dropna()
    closes = {pd.Timestamp(d).date().isoformat():float(v) for d,v in spy.items() if math.isfinite(float(v)) and float(v)>0}
    by_day = {}
    for row in sorted(rows, key=lambda r: r['published_at']):
        stamp = pd.Timestamp(row['published_at'])
        if stamp.tzinfo is None: stamp = stamp.tz_localize('UTC')
        day = stamp.tz_convert('America/New_York').date().isoformat()
        try: score = float(row['score'])
        except (ValueError, TypeError): continue
        if not math.isfinite(score) or not 0 <= score <= 100: continue
        if day not in closes or day in by_day: continue
        by_day[day] = {'date':day,'score':score,'spy_close':closes[day],
                       'published_at':stamp.isoformat(),'run_id':str(row['id'])}
    points = list(by_day.values())
    if points:
        base=points[0]['spy_close']
        for p in points: p['spy_index']=round(100*p['spy_close']/base,3)
    return points
