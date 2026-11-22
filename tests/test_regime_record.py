import pandas as pd
from api.services.regime_record import compare_published

def test_record_uses_publication_date_first_publication_and_adjusted_close():
    rows=[{'id':'a','published_at':'2026-11-20T20:00:00Z','score':'23'},
          {'id':'b','published_at':'2026-11-20T21:00:00Z','score':'80'},
          {'id':'c','published_at':'2026-11-23T20:00:00Z','score':'24'},
          {'id':'d','published_at':'2026-11-24T20:00:00Z','score':'invalid'}]
    prices=pd.DataFrame({'SPY':[100,110]},index=pd.to_datetime(['2026-11-20','2026-11-23']))
    out=compare_published(rows,prices)
    assert [p['score'] for p in out]==[23,24]
    assert [p['spy_index'] for p in out]==[100,110]
    assert out[0]['run_id']=='a'
def test_missing_market_date_is_not_forward_filled():
    rows=[{'id':'a','published_at':'2026-11-21T20:00:00Z','score':23}]
    prices=pd.DataFrame({'SPY':[100]},index=pd.to_datetime(['2026-11-20']))
    assert compare_published(rows,prices)==[]
