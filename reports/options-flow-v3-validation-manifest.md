# Signal Validation v3 -- Frozen Pre-Registration Manifest

**Written and committed BEFORE any new historical data was pulled.** Its purpose is to make
after-the-fact reinterpretation impossible: everything below is a verbatim copy of what commit
`19dc02c` already computed and already claimed, not a fresh look at the idea.

- **Discovery commit:** `19dc02c` (Signal Design v2)
- **Methodology version:** `1.0.0` (unchanged)
- **Primary feature under validation:** `z_gross_premium_20d`

## Frozen definition (verbatim from commit 19dc02c)

```python
Z_WINDOW, Z_MIN_PERIODS = 20, 10

def _zscore(col: str) -> pd.Series:
    prior = g[col].shift(1)
    roll_mean = prior.groupby(df["ticker"]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    roll_std = prior.groupby(df["ticker"]).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
    roll_mean.index = roll_mean.index.droplevel(0)
    roll_std.index = roll_std.index.droplevel(0)
    z = (df[col] - roll_mean) / roll_std
    return z.replace([np.inf, -np.inf], np.nan)

df["z_gross_premium_20d"] = _zscore("gross_premium")
```

- **Rolling window:** 20 trading sessions.
- **Minimum prior observations:** 10 (fewer -> `NaN`, never a fabricated 0).
- **Is date T excluded from its own baseline?** Yes -- `.shift(1)` is applied before `.rolling()`,
  so T's z-score never uses T's own gross premium.
- **Standard deviation convention:** pandas default, sample std (`ddof=1`).
- **Grouping:** strictly per-ticker. Each ETF is compared only to its own trailing history,
  never pooled across ETFs.
- **Null handling:** `NaN` during warmup or when the trailing window has zero variance; `inf`/
  `-inf` (a variance-collapse edge case) is also mapped to `NaN`. Never zero-filled.
- **Winsorization:** none.

## Expected direction, recorded before touching new data

**Sign: POSITIVE.** Elevated `z_gross_premium_20d` (today's gross options premium is unusually
high relative to that ETF's own trailing 20-session baseline) is expected to be followed by
**higher** forward returns. This sign will not be flipped after seeing the validation sample,
regardless of what it shows.

### v2 discovery-sample result (60 sessions x 6 ETFs, 2026-06-26 to 2026-09-21, n=300 after warmup)

| Horizon | N | Q1 mean | Q1 HAC-t | Q5 mean | Q5 HAC-t | Q5-Q1 |
|---|---:|---:|---:|---:|---:|---:|
| 1D | 300 | -0.07% | -0.39 | 0.31% | 1.73 | 0.38% |
| 3D | 300 | -0.63% | -1.86 | 0.79% | 3.02 | 1.42% |
| 5D | 300 | -0.84% | -2.13 | 1.20% | 2.82 | 2.04% |
| 10D | 300 | -1.45% | -4.28 | 2.23% | 2.60 | 3.68% |
| 20D | 300 | -1.36% | -3.12 | 2.31% | 3.29 | 3.67% |

Spearman(quintile, 10D return): rho=0.332, p=6.9e-08, n=252. Per-ETF sign at 5D/10D: all 6 of 6
ETFs positive. **Caveat already on record in v2:** the Q5 bucket's 60 rows span only 34 unique
calendar dates (~1.76 rows/date) -- some of that "6/6 agreement" may reflect shared macro-event
dates rather than independent confirmation. Resolving exactly this ambiguity is v3's job.

## Sample composition

- **Discovery sample:** the existing 60 full_flow sessions x 6 ETFs (already committed, source
  of the hypothesis). Reported separately. **Never counted toward the primary validation
  significance test.**
- **Validation sample (target):** 192 new, previously unseen full_flow sessions per ETF,
  immediately preceding the discovery window on the trading calendar.
- **Feature warmup (target):** 20 additional full_flow sessions per ETF, immediately preceding
  the validation sample, used only to seed each ETF's own trailing-20D baseline for the earliest
  validation dates. **Not counted as validation observations themselves.**
- **Total new full_flow pull (target):** ~212 sessions per ETF (192 + 20), computed from the
  real trading calendar, not by calendar-day subtraction from 2026-06-26.
- **Combined sample (descriptive only):** 60 + 192 = 252 sessions. Never presented as if all 252
  were out-of-sample.

## Forward horizons

- **Primary:** 5D, 10D, 20D.
- **Secondary (reported for completeness only):** 1D, 3D.

## Explicitly prohibited during primary validation

Changing the 20-session window; trying 10D/30D/60D alternatives; optimizing any cutoff; adding
interaction features; changing forward-return horizons; cherry-picking ETFs; dropping
bad-performing dates; flipping the recorded sign; training an ML model; and calculating any v3
result before the full validation backfill is complete (no incremental peeking).
