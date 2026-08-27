import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import regime, assets, signals, charts, stocks
from api.routers.stock_intelligence import router as stock_intelligence_router
from api.routers.ema_inflection import router as ema_inflection_router
from api.routers.stock_rankings import router as stock_rankings_router
from api.routers.stock_portfolio import router as stock_portfolio_router
from api.routers.smid_growth_portfolio import router as smid_growth_portfolio_router
from api.routers.portfolio_snapshots import router as portfolio_snapshots_router
from api.routers.volatility import router as volatility_router
from api.routers.stock_intelligence_snapshots import router as stock_intelligence_snapshots_router


try:
    from api.routers import portfolio
except ImportError:
    portfolio = None
try:
    from api.routers import screener
except ImportError:
    screener = None
try:
    from api.routers import playbook
except ImportError:
    playbook = None

app = FastAPI(title="Macro Engine API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.macro-engine.com","https://macro-engine.com",
                   "http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["GET"], allow_headers=["*"],
)

app.include_router(regime.router,   prefix="/api")
app.include_router(assets.router,   prefix="/api")
app.include_router(signals.router,  prefix="/api")
app.include_router(charts.router,   prefix="/api")
app.include_router(stocks.router,   prefix="/api")
app.include_router(stock_intelligence_router)
app.include_router(ema_inflection_router)
app.include_router(stock_rankings_router)
app.include_router(stock_portfolio_router)
app.include_router(smid_growth_portfolio_router)
app.include_router(portfolio_snapshots_router)
app.include_router(volatility_router)
app.include_router(stock_intelligence_snapshots_router)
if portfolio:  app.include_router(portfolio.router, prefix="/api")
if screener:   app.include_router(screener.router,  prefix="/api")
if playbook:   app.include_router(playbook.router,  prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "Macro Engine API", "version": "1.3.0", "docs": "/docs"}