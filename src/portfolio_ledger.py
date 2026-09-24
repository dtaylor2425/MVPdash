"""Pure, append-only quantitative portfolio accounting; no network or database IO.

Published history is copied verbatim. New decisions execute at the next session
open AFTER publication day, never on the bar that generated the decision.
Prices are raw USD listing prices; split/dividend events are booked explicitly.
"""
from copy import deepcopy
from datetime import date, timedelta
import math
import statistics

VERSION = "published-ledger-v1"


def number(value, default=None):
    if isinstance(value, bool):
        return default
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def macro_target(score):
    score = number(score)
    if score is None or not 0 <= score <= 100:
        raise ValueError("A valid macro score is required to publish allocation changes")
    return round(0.65 + 0.30 * min(1.0, max(0.0, (score - 25.0) / 50.0)), 6)


def volume_signal(bars):
    """Latest completed bar against PRIOR 20 sessions (no current-bar dilution)."""
    if len(bars) < 21:
        return {"relative_volume": None, "volume_signal": "Volume unavailable", "volume_multiplier": 1.0}
    volumes = [number(b.get("signal_volume", b.get("volume"))) for b in bars[-21:-1]]
    latest = bars[-1]
    latest_volume = number(latest.get("signal_volume", latest.get("volume")))
    if any(v is None or v <= 0 for v in volumes) or latest_volume is None or latest_volume <= 0:
        return {"relative_volume": None, "volume_signal": "Volume unavailable", "volume_multiplier": 1.0}
    ratio = latest_volume / (sum(volumes) / 20)
    current = latest.get("signal_close", latest["close"])
    prior = bars[-2].get("signal_close", bars[-2]["close"])
    up = current > prior
    down = current < prior
    label, multiplier = "Normal participation", 1.0
    if ratio >= 1.5 and up:
        label, multiplier = "Strong buying volume", 1.15
    elif ratio >= 1.5 and down:
        label, multiplier = "Strong selling volume", 0.65
    elif ratio < 0.8 and up:
        label, multiplier = "Weak buying volume", 0.85
    elif ratio < 0.8 and down:
        label, multiplier = "Weak selling volume", 0.95
    return {"relative_volume": round(ratio, 4), "volume_signal": label, "volume_multiplier": multiplier}


def first_entries(published):
    """First published entry in the CURRENT uninterrupted holding period.

    Do not infer dates/prices from a recalculated backtest or current quote.
    Existing stored entry metadata always wins over the snapshot close.
    """
    entries = {}
    for run in sorted(published, key=lambda r: (str(r["run_date"]), str(r.get("published_at") or ""))):
        rows = run.get("payload", run).get("holdings", [])
        present = {r["ticker"] for r in rows}
        entries = {t: e for t, e in entries.items() if t in present}
        for row in rows:
            ticker = row["ticker"]
            if ticker not in entries:
                recorded = number(row.get("entry_price"))
                if recorded is None:
                    recorded = number(row.get("close"), number(row.get("price")))
                entries[ticker] = {
                    "entry_date": row.get("entry_date") or str(run.get("published_at") or run["run_date"])[:10],
                    "entry_price": recorded,
                    "entry_source_run_id": str(run.get("id") or run.get("run_id") or ""),
                    "entry_price_status": "first_published_reference" if recorded else "legacy_unrecorded",
                    "entry_source": "First published model record; not a broker fill",
                    "entry_model_date": str(run["run_date"])[:10],
                    "entry_published_at": str(run.get("published_at") or ""),
                }
    return entries


def target_holdings(candidate, market, score):
    """Respect stock/sector caps; broad SPY sleeve fills unused long budget."""
    target = macro_target(score)
    rows = deepcopy(candidate.get("holdings") or [])
    if not rows:
        raise ValueError("No candidate holdings; refusing to replace published portfolio")
    for row in rows:
        row.update(volume_signal(market.get(row["ticker"], [])))
        row["_allocation"] = max(0.0, number(row.get("target_weight"), 0)) * row["volume_multiplier"]
    total = sum(r["_allocation"] for r in rows)
    if total <= 0:
        raise ValueError("Candidate weights are missing")
    rules = candidate.get("risk_rules") or {}
    cap = number(rules.get("max_single_position"), 0.12)
    sector_cap = number(rules.get("max_sector_weight"), 1.0)
    theme_cap = number(rules.get("max_theme_weight"), 1.0)
    for row in rows:
        row["target_weight"] = min(cap, target * row.pop("_allocation") / total)
    for key, limit in (("sector", sector_cap), ("theme", theme_cap)):
        groups = {}
        for row in rows:
            groups.setdefault(row.get(key) or "Unknown", []).append(row)
        for group in groups.values():
            weight = sum(r["target_weight"] for r in group)
            if weight > limit:
                for row in group:
                    row["target_weight"] *= limit / weight
    reserve = target - sum(r["target_weight"] for r in rows)
    if reserve > 1e-8:
        existing = next((r for r in rows if r["ticker"] == "SPY"), None)
        if existing:
            existing["target_weight"] += reserve
        else:
            rows.append({"ticker": "SPY", "name": "S&P 500 long-exposure sleeve", "sector": "Diversified ETF",
                         "theme": "Macro long-exposure reserve", "target_weight": reserve,
                         "volume_signal": "Allocation reserve", "relative_volume": None,
                         "trade_reason": "Maintains macro long allocation without relaxing individual-stock concentration limits."})
    return rows, target


def _bar(market, ticker, day):
    row = next((b for b in market.get(ticker, []) if b["date"] == day), None)
    if row is None or number(row.get("close"), 0) <= 0 or number(row.get("open"), 0) <= 0:
        raise ValueError(f"Missing valid market bar for {ticker} on {day}; published ledger unchanged")
    return row


def _views(book):
    nav = book["cash"] + sum(p["quantity"] * p["current_price"] for p in book["positions"].values())
    rows = []
    for ticker, p in book["positions"].items():
        row = deepcopy(p)
        row["ticker"] = ticker
        row["current_weight"] = p["quantity"] * p["current_price"] / nav
        entry = number(p.get("entry_price"), 0)
        row["unrealized_return_pct"] = (p["current_price"] / entry - 1) * 100 if entry > 0 else None
        row["cost_basis_complete"] = entry > 0
        rows.append(row)
    return sorted(rows, key=lambda r: r["current_weight"], reverse=True), nav


def build_ledger(candidate, previous, published, market, run_date, macro_score):
    """Build a new publication from stored state, never from rebuilt performance."""
    run_day = str(run_date)[:10]
    sessions = sorted(b["date"] for b in market.get("SPY", []) if b["date"] < run_day)
    if not sessions:
        raise ValueError("No completed benchmark session")
    out = deepcopy(candidate)
    old = previous.get("payload", {}) if previous else {}
    if not previous:
        raise ValueError("Existing published portfolio required; initial launch needs an explicit opening record")
    performance = deepcopy(old.get("performance") or {"series": [], "stats": {}, "rebalance_log": []})
    book = deepcopy(old.get("ledger"))
    trades = []
    if book:
        for ticker, position in book["positions"].items():
            reference = _bar(market, ticker, book["as_of"])["close"]
            if not math.isclose(reference, position["current_price"], rel_tol=1e-6, abs_tol=.0001):
                raise ValueError(f"Provider revised the last published price for {ticker}; reconciliation required")
    if not book:
        # Preserve the old curve, but do NOT invent quantities/trades before the
        # first accounting anchor. Existing entries are documented references.
        last = (performance.get("series") or [{}])[-1]
        anchor = sessions[-1]
        if not last.get("date"):
            raise ValueError("Legacy performance has no dated accounting anchor")
        entries = first_entries(published)
        positions = {}
        holdings = old.get("holdings") or []
        total = sum(number(r.get("target_weight"), 0) for r in holdings)
        if total > 1.000001:
            raise ValueError("Legacy weights exceed 100%; reconcile before migrating")
        for row in holdings:
            ticker = row["ticker"]
            price = _bar(market, ticker, anchor)["close"]
            weight = number(row.get("target_weight"), 0)
            if weight > 0:
                positions[ticker] = {**deepcopy(row), **entries.get(ticker, {}), "quantity": weight / price,
                                     "current_price": price}
        last = (performance.get("series") or [{}])[-1]
        book = {"version": VERSION, "anchor_date": anchor, "as_of": anchor, "cash": 1-total,
                "positions": positions, "base_model": number(last.get("model"), 0),
                "base_benchmark": number(last.get("benchmark"), 0),
                "benchmark_anchor": _bar(market, "SPY", anchor)["close"], "benchmark_units": 1.0,
                "benchmark_cash": 0.0, "pending": None, "trades": [], "last_decision_date": None}
        book["legacy_through"] = str(last["date"])[:10]
        book["legacy_rebalance_count"] = number((performance.get("stats") or {}).get("rebalance_count"), 0)
        if last.get("date") and str(last["date"])[:10] > anchor:
            raise ValueError("Published history is ahead of completed market data")
    if book:
        for day in sessions:
            if day <= book["as_of"]:
                continue
            # Corporate actions change units/cash, not historical published values.
            for ticker, position in book["positions"].items():
                bar = _bar(market, ticker, day)
                split = number(bar.get("split"), 0)
                if split > 0 and split != 1:
                    position["quantity"] *= split
                    if number(position.get("entry_price"), 0) > 0:
                        position.setdefault("original_entry_price", position["entry_price"])
                        position["entry_price"] /= split
                book["cash"] += position["quantity"] * number(bar.get("dividend"), 0)
            bench = _bar(market, "SPY", day)
            split = number(bench.get("split"), 0)
            if split > 0:
                book["benchmark_units"] *= split
            book["benchmark_cash"] += book["benchmark_units"] * number(bench.get("dividend"), 0)
            pending = book.get("pending")
            if pending and day >= pending["execute_not_before"]:
                opening_nav = book["cash"] + sum(p["quantity"] * _bar(market, t, day)["open"] for t, p in book["positions"].items())
                positions = {}
                targets = {r["ticker"]: r for r in pending["holdings"]}
                for ticker in sorted(set(book["positions"]) | set(targets)):
                    bar = _bar(market, ticker, day)
                    prior = book["positions"].get(ticker)
                    target_row = targets.get(ticker)
                    quantity = opening_nav * target_row["target_weight"] / bar["open"] if target_row else 0
                    delta = quantity - (prior["quantity"] if prior else 0)
                    if abs(delta) > 1e-12:
                        trade = {"date": day, "ticker": ticker, "action": "Buy" if delta > 0 else "Sell",
                                       "quantity": abs(delta), "price": bar["open"], "decision_date": pending["decision_date"],
                                       "weight_traded": abs(delta) * bar["open"] / opening_nav,
                                       "price_basis": "Next-session opening model reference"}
                        if delta < 0:
                            entry_price = number((prior or {}).get("entry_price"), 0)
                            trade.update({"exit_type": "Full exit" if quantity <= 0 else "Trim",
                                          "entry_date": (prior or {}).get("entry_date"),
                                          "entry_price": entry_price if entry_price > 0 else None,
                                          "entry_source": (prior or {}).get("entry_source"),
                                          "exit_price": bar["open"],
                                          "exit_return_pct": (bar["open"] / entry_price - 1) * 100 if entry_price > 0 else None,
                                          "return_basis": "Price change from the recorded model entry, split-adjusted; excludes dividends, fees and taxes. Not average-cost realized P/L."})
                        trades.append(trade)
                    if target_row and quantity > 0:
                        entry = {k: prior[k] for k in ("entry_date", "entry_price", "entry_source", "entry_source_run_id", "entry_price_status", "original_entry_price", "entry_model_date", "entry_published_at") if prior and k in prior}
                        if not prior:
                            entry = {"entry_date": day, "entry_price": bar["open"], "entry_source": "Next-session opening model reference",
                                     "entry_price_status": "recorded_model_entry"}
                        positions[ticker] = {**deepcopy(target_row), **entry, "quantity": quantity, "current_price": bar["close"]}
                book["positions"] = positions
                book["cash"] = opening_nav * (1 - sum(r["target_weight"] for r in targets.values()))
                book["pending"] = None
            for ticker, p in book["positions"].items():
                p["current_price"] = _bar(market, ticker, day)["close"]
            _, nav = _views(book)
            point = {"date": day, "model": ((1+book["base_model"]) * nav - 1),
                     "benchmark": (1+book["base_benchmark"]) * (book["benchmark_units"]*bench["close"]+book["benchmark_cash"]) / book["benchmark_anchor"] - 1}
            series = performance.setdefault("series", [])
            if series and day <= str(series[-1]["date"])[:10]:
                raise ValueError("Cannot append a valuation over published history")
            series.append(point)
            book["as_of"] = day
    book["trades"].extend(trades)
    for ticker, position in book["positions"].items():
        position.update(volume_signal([b for b in market.get(ticker, []) if b["date"] <= book["as_of"]]))
    holdings, nav = _views(book)
    completed_market = {t: [b for b in bars if b["date"] < run_day] for t, bars in market.items()}
    target_rows, exposure = target_holdings(candidate, completed_market, macro_score)
    old_target = number(old.get("target_stock_exposure"), exposure)
    last_decision = book.get("last_decision_date")
    week_changed = not last_decision or date.fromisoformat(last_decision).isocalendar()[:2] != date.fromisoformat(run_day).isocalendar()[:2]
    invested = 1-book["cash"]/nav
    if book.get("pending") is None and (week_changed or invested < .65 or abs(exposure-old_target) >= .05):
        book["pending"] = {"decision_date": run_day, "execute_not_before": (date.fromisoformat(run_day)+timedelta(days=1)).isoformat(),
                           "holdings": target_rows, "macro_score": macro_score, "target_stock_exposure": exposure}
        book["last_decision_date"] = run_day
    if book.get("pending"):
        exposure = book["pending"]["target_stock_exposure"]
    else:
        exposure = old_target
    out.update({"ledger": book, "holdings": holdings, "stock_exposure": invested, "cash_weight": book["cash"]/nav,
                "target_stock_exposure": exposure, "valuation_as_of": book["as_of"], "macro_score": macro_score,
                "exposure_regime": "Macro allocation: 65–95% long target", "performance": performance,
                "trade_queue": trades, "pending_allocation": book.get("pending")})
    for key, group in (("sector_weights", "sector"), ("theme_weights", "theme")):
        out[key] = {}
        for row in holdings:
            label = row.get(group) or "Unknown"
            out[key][label] = out[key].get(label, 0) + row["current_weight"]
    stats = performance.setdefault("stats", {})
    if performance.get("series"):
        stats["model_return"] = performance["series"][-1]["model"]
        stats["benchmark_return"] = performance["series"][-1]["benchmark"]
        for key in ("model", "benchmark"):
            values = [1 + number(p.get(key), 0) for p in performance["series"]]
            returns = [b/a - 1 for a, b in zip(values, values[1:]) if a > 0]
            peak, drawdown = values[0], 0.0
            for value in values:
                peak = max(peak, value)
                if peak > 0:
                    drawdown = min(drawdown, value/peak - 1)
            stats[key + "_max_drawdown"] = drawdown
            stats[key + "_volatility"] = statistics.stdev(returns)*math.sqrt(252) if len(returns) > 1 else None
        stats["rebalance_count"] = book.get("legacy_rebalance_count", 0) + len({t["decision_date"] for t in book["trades"]})
    performance["methodology_note"] = (
        f"Quantitative model portfolio. Earlier model history uses retrospective inputs; published values are retained unchanged. "
        f"Prospective accounting starts {book['anchor_date']}. New decisions use the next session open after publication day. "
        f"Legacy history ends {book.get('legacy_through', book['anchor_date'])}; any gap to the accounting start is unmeasured and adds no return. "
        "Model reference prices are not broker fills; returns exclude transaction costs and taxes. "
        "The 65–95% long range is a rebalance target; prices can move actual exposure outside it between executions."
    )
    out.setdefault("methodology", {}).update({"volume": "Prior 20-session relative volume: strong >=1.5x, weak <0.8x; price direction is a buying/selling-pressure proxy, not observed order flow. Allocation multipliers 1.15/0.65/0.85/0.95 respectively.",
        "macro_allocation": "65% at scores <=25, rising linearly to 95% at >=75. Broad-market SPY fills capacity left by individual-stock caps.",
        "entry": "First published entry reference for the current holding period; adds/trims do not reset it. Price change since entry is not a money-weighted position return."})
    return out
