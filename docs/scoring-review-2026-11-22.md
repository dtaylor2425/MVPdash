# Scoring review — November 22, 2026

The overall macro model uses six components: Growth 25%, Inflation 15%, Monetary policy 20%, Market internals 20%, Fiscal/external 10%, Sentiment 10%. Under stress these become 15/8/20/37/10/10. Normalization and divergence adjustments also affect the final score. No formula changes in this release.

Growth, Inflation, Rates and Credit pages each use independent frontend threshold rubrics. They are not the same scores as the backend components. Fixed navigation weights were removed and these scores are now labeled page diagnostics. Volatility shows raw indices, not an independent 20%-weighted score.

FX uses five cross-sectional components with missing-input reweighting: carry/policy momentum 31.25% each, terms of trade/trend/valuation 12.5% each. Stock workbook research-v2 uses profitability/growth/balance/valuation/cash weights of 25/25/20/15/15, requiring three components and 60% coverage. Asset signals use four 25-point dimensions. Stock-selection and portfolio diagnostics are separate from these systems and from actual ledger returns.

## Recorded comparison
The dashboard now compares macro scores saved in first-published daily Alpha snapshots against matching-date Yahoo adjusted SPY closes rebased to 100. Publication timestamps and run IDs are exposed for traceability. No recalculated historical scores or forward-filled missing market dates are substituted. The graph is a contemporaneous comparison, not a strategy return or proof of prediction. SPY closes can follow publication, and historical model versions may differ.

## Outstanding findings
- src/regime.py compares HY OAS against 500 despite FRED returning percentage points; 5.0 appears intended for a 500bp trigger. VIX >28 independently activates stress. Correct through a versioned change after comparing its impact on portfolio decisions.
- Model comments claim 70/30 persistence, but the public computation calls the core without that smoothing. Comments also overstate exact weight doubling and understate the maximum divergence deduction.
- Growth/internals raw zero signals can be excluded from active weights; neutral and unavailable signals should be distinguished.
- Page diagnostics begin at 50 and do not have adequate partial-coverage checks. They are not calibrated probabilities.
- Existing /regime/history recalculates using current FRED vintages and observation dates, not historical release-time availability. It is not a published track record.
- Future predictive evaluation requires fixed-time, versioned observations, subsequent-session returns, sample sizes and untouched holdout data.

Validation: publication-date matching, first-publication retention, adjusted-price rebasing and missing-date tests passed. Frontend production build checked before publishing.
