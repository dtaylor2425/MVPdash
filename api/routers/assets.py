"""
api/routers/assets.py
GET /api/assets — scored asset alignment using four-factor model
"""

from fastapi import APIRouter, HTTPException

from api.deps import get_macro, get_prices, get_regime

router = APIRouter(tags=["Assets"])


@router.get("/assets")
def assets_scored():
    try:
        macro = get_macro()
        px = get_prices()
        regime = get_regime()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        from src.asset_scoring import score_all_assets
        results = score_all_assets(
            macro=macro,
            proxies=px,
            regime_score=regime.score,
            regime_label=regime.label,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Asset scoring error: {}".format(str(e)))

    return {"assets": results}
#comment