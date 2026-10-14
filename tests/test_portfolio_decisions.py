from copy import deepcopy
import pytest
from src.portfolio_decisions import daily_targets

RULES={"max_single_position":.12,"max_sector_weight":.34}
def holding(t,w,stop=None):
 return {"ticker":t,"current_weight":w,"target_weight":w,"sector":"Tech" if t!="SPY" else "ETF","stop_level":stop}
def bars(p): return [{"date":"2026-10-08","close":p}]

def test_stop_exit_even_if_desired_still_wants_stock_and_macro_unchanged():
 old=[holding("A",.1,95),holding("B",.1),holding("SPY",.54)]
 desired=deepcopy(old); before=deepcopy(old)
 rows,review=daily_targets(old,desired,.74,{"A":bars(90),"B":bars(100)},RULES)
 assert "A" not in [r["ticker"] for r in rows]
 assert review["risk_exits"] == {"A":"Completed close below published stop"}
 assert sum(r["target_weight"] for r in rows)==pytest.approx(.74)
 assert old==before

def test_small_changes_do_not_force_cosmetic_trades():
 old=[holding("A",.1),holding("SPY",.64)]
 desired=[holding("A",.105),holding("SPY",.635)]
 assert daily_targets(old,desired,.74,{"A":bars(100)},RULES)[0] is None

def test_large_rotation_staged_with_per_name_and_total_budget():
 old=[holding("A",.1),holding("B",.1),holding("SPY",.54)]
 desired=[holding("C",.12),holding("D",.12),holding("SPY",.5)]
 rows,review=daily_targets(old,desired,.74,{"A":bars(100),"B":bars(100)},RULES)
 oldw={r["ticker"]:r["current_weight"] for r in old}; neww={r["ticker"]:r["target_weight"] for r in rows}
 assert all(abs(neww.get(t,0)-oldw.get(t,0)) <= .0200001 for t in set(neww)|set(oldw) if t!="SPY")
 assert sum(abs(neww.get(t,0)-oldw.get(t,0)) for t in set(neww)|set(oldw))/2 <= .100001
 assert sum(neww.values())==pytest.approx(.74)

def test_published_stop_does_not_reset_down_on_trim():
 old=[holding("A",.1,95),holding("SPY",.64)]
 desired=[holding("A",.07,85),holding("SPY",.67)]
 rows,_=daily_targets(old,desired,.74,{"A":bars(100)},RULES)
 assert next(r for r in rows if r["ticker"]=="A")["stop_level"]==95

def test_falling_trend_exit_without_stop():
 old=[holding("A",.1),holding("SPY",.64)]
 rows,review=daily_targets(old,old,.74,{"A":[{"close":100-i} for i in range(30)]},RULES)
 assert "A" in review["risk_exits"]


def test_risk_exit_cooldown_blocks_immediate_reentry():
 old=[holding("SPY",.74)]
 rows,review=daily_targets(old,[holding("A",.12),holding("SPY",.62)],.74,{},RULES,blocked=["A"])
 assert all(r["ticker"]!="A" for r in (rows or []))
