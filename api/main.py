"""
api/main.py
═══════════════════════════════════════════════════════════════════════════════
Macro Engine FastAPI backend.

Exposes your existing Python engine (src/regime.py, src/data_sources.py, etc.)
as a REST API so the Next.js frontend can consume live data.

Run locally:
    uvicorn api.main:app --reload --port 8000

Deploy on Railway:
    Start command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

import sys
import os

# Make sure src/ is importable from anywhere
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import regime, assets, signals, charts

app = FastAPI(
    title="Macro Engine API",
    description="Quantitative macro regime scoring and asset signals.",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow requests from your Next.js frontend and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.macro-engine.com",
        "https://macro-engine.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(regime.router,  prefix="/api")
app.include_router(assets.router,  prefix="/api")
app.include_router(signals.router, prefix="/api")
app.include_router(charts.router,  prefix="/api")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "name": "Macro Engine API",
        "version": "1.0.0",
        "docs": "/docs",
    }