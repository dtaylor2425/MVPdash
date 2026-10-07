"""Versioned, coverage-aware company evidence rubric for the stock workbook.

This does not change published ranking or portfolio selection formulas.
"""
from __future__ import annotations
import math
import statistics


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def scale(value, low, high):
    return max(0.0, min(100.0, (value - low) / (high - low) * 100))


def build_research_score(info):
    components = []
    financial = str(info.get("sector", "")).lower() in {"financial services", "financials"}
    def component(key, label, weight, inputs, explanation):
        available = [i for i in inputs if i[1] is not None]
        components.append({"key":key,"label":label,"weight":weight,
            "value":round(statistics.mean(i[2](i[1]) for i in available),1) if available else None,
            "explanation":explanation,
            "inputs":[{"label":i[0],"value":i[1],"unit":i[3]} for i in inputs]})
    op_margin = number(info.get("operatingMargins"))
    roa = number(info.get("returnOnAssets"))
    roe = number(info.get("returnOnEquity"))
    component("profitability","Profitability",25,
        [("Return on equity",roe,lambda x:scale(x,0,.20),"fraction")] if financial else
        [("Operating margin",op_margin,lambda x:scale(x,0,.30),"fraction"),("Return on assets",roa,lambda x:scale(x,0,.15),"fraction")],
        "Operating margin: 0–30%; return on assets: 0–15%. Financial companies instead use return on equity: 0–20%. Available inputs are equally weighted.")
    growth = number(info.get("revenueGrowth"))
    component("growth","Revenue growth",25,[("Reported revenue growth",growth,lambda x:scale(x,-.10,.40),"fraction")],
        "Reported revenue growth is mapped from -10% (0 points) to +40% (100 points). This is a fixed rubric, not an industry percentile.")
    current = number(info.get("currentRatio")) if not financial else None
    debt = number(info.get("debtToEquity")) if not financial else None
    # Negative book equity is not evidence of a strong balance sheet.
    debt = debt if debt is None or debt >= 0 else None
    component("balance","Balance sheet",20,[("Current ratio",current,lambda x:scale(x,.5,2.5),"multiple"),("Debt / equity",debt,lambda x:100-scale(x,0,200),"percent")],
        "Current ratio: 0.5–2.5; debt/equity: 200%–0%. Excludes negative-equity ratios and financial companies, whose leverage needs a different framework.")
    pe = number(info.get("forwardPE"))
    pe = pe if pe is not None and pe > 0 else None
    component("valuation","Valuation discipline",15,[("Forward P/E",pe,lambda x:100-scale(x,10,50),"multiple")],
        "Positive forward P/E of 10x or less receives 100 points; 50x or more receives 0. Based on estimates, not a fair-value target; not sector-normalized.")
    revenue,fcf = number(info.get("totalRevenue")),number(info.get("freeCashflow"))
    margin = fcf / revenue if revenue is not None and revenue > 0 and fcf is not None and not financial else None
    component("cash","Cash generation",15,[("Free cash flow / revenue",margin,lambda x:scale(x,0,.25),"fraction")],
        "Free cash flow divided by revenue: 0–25%. Both are provider-reported financial-currency amounts. Not applied to financial companies.")
    weight = sum(c["weight"] for c in components if c["value"] is not None)
    count = sum(c["value"] is not None for c in components)
    total = sum(c["value"]*c["weight"] for c in components if c["value"] is not None)/weight if weight >= 60 and count >= 3 else None
    return {"version":"research-v2","total":round(total,1) if total is not None else None,
        "coverage_pct":weight,"label":"Insufficient coverage" if total is None else "Strong evidence" if total >= 70 else "Mixed evidence" if total >= 40 else "Weak evidence",
        "components":components,"methodology":[
            "Company evidence score uses five disclosed fundamental rubrics. It is separate from price participation, published stock rankings, and portfolio construction.",
            "Missing data is never assigned 50. Available pillar weights are rescaled; at least three pillars and 60% weight must be available.",
            "Bounds are research design choices, not calibrated return probabilities or sector-relative ranks. A high score does not imply an attractive entry price.",
            "Source: Yahoo Finance company fundamentals and analyst forward earnings estimates. ThetaData supplies the separate market participation panel."]}
