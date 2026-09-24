# Macro Options Flow -- Phase 1 backfill QA report

Generated 2026-09-23T21:38:14+00:00 UTC. Methodology version `1.0.0`. Status: **COMPLETE**.

## Item 5: per-ETF and combined summary

| Ticker | Full-flow req | S | P | F | Warmup req | S | P | F | Total Theta req | Median req/day | Median runtime/day (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 60 | 60 | 0 | 0 | 252 | 244 | 8 | 0 | 6,645 | 18 | 42.0 |
| QQQ | 60 | 60 | 0 | 0 | 252 | 228 | 24 | 0 | 6,621 | 18 | 41.6 |
| IWM | 60 | 60 | 0 | 0 | 252 | 252 | 0 | 0 | 6,445 | 18 | 37.4 |
| SMH | 60 | 60 | 0 | 0 | 252 | 248 | 4 | 0 | 3,805 | 8 | 16.3 |
| TLT | 60 | 60 | 0 | 0 | 252 | 252 | 0 | 0 | 5,039 | 14 | 27.6 |
| GLD | 60 | 59 | 1 | 0 | 252 | 248 | 4 | 0 | 5,191 | 14 | 27.9 |
| ALL | 360 | 359 | 1 | 0 | 1512 | 1472 | 40 | 0 | 33,746 | 17 | 34.2 |

Total wall-clock time (sum of runtimeSec across all ticker-days): 74532s (20.7h)

### Quality medians (full-flow only) and missing counts

| Ticker | Trades | Eligible | Gross premium | Classified | Greek match | OI match | Missing ATM | 7D | 30D | 60D | Skew | Delta | 20D% | 60D% | 126D% | 252D% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 1429500 | 1429366 | $1,341,642,345 | 100.0% | 100.0% | 99.2% | 0 | 0 | 0 | 3 | 1 | 0 | 14 | 44 | 94 | 189 |
| QQQ | 1034594 | 1034300 | $1,276,307,328 | 100.0% | 100.0% | 99.3% | 0 | 0 | 0 | 3 | 12 | 0 | 14 | 44 | 94 | 189 |
| IWM | 166348 | 166348 | $155,553,794 | 100.0% | 100.0% | 98.5% | 0 | 0 | 0 | 9 | 0 | 0 | 14 | 44 | 94 | 189 |
| SMH | 20008 | 20005 | $119,275,300 | 100.0% | 100.0% | 97.7% | 0 | 0 | 0 | 11 | 0 | 0 | 14 | 44 | 94 | 189 |
| TLT | 21308 | 21308 | $39,269,084 | 100.0% | 100.0% | 93.1% | 0 | 0 | 0 | 7 | 0 | 0 | 14 | 44 | 94 | 189 |
| GLD | 40379 | 40376 | $60,371,536 | 100.0% | 100.0% | 97.7% | 0 | 0 | 0 | 9 | 0 | 0 | 14 | 44 | 94 | 189 |
| ALL | 96954 | 96954 | $155,074,932 | 100.0% | 100.0% | 98.4% | 0 | 0 | 0 | 42 | 13 | 0 | 84 | 264 | 564 | 1134 |

## Item 6: distribution QA (full-flow, per ETF)

### SPY
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.0668 | -0.0628 | -0.0453 | -0.0054 | -0.0069 | 0.0322 | 0.0501 | 0.0532 | 60 |
| delta_imbalance_ratio | -0.0337 | -0.0281 | -0.0220 | -0.0011 | 0.0003 | 0.0276 | 0.0364 | 0.0404 | 60 |
| net_dollar_delta_imbalance | -7568548906.4271 | -6594339895.5374 | -5502619422.0843 | -280861402.2961 | 141333144.2540 | 7595780773.2535 | 10211342559.7407 | 12309747981.8978 | 60 |
| gross_premium | 803982168.0000 | 806515828.6000 | 878993910.4000 | 1341642345.0000 | 1392861614.6500 | 2021171699.8000 | 2631788789.1200 | 2770745254.0000 | 60 |
| atm_iv | 0.1157 | 0.1168 | 0.1183 | 0.1315 | 0.1345 | 0.1559 | 0.1659 | 0.1752 | 60 |
| put_skew_25d | 0.0267 | 0.0269 | 0.0280 | 0.0471 | 0.0460 | 0.0613 | 0.0628 | 0.0645 | 59 |
| zero_dte_share | 0.3168 | 0.3268 | 0.3434 | 0.4093 | 0.4116 | 0.4844 | 0.5037 | 0.5065 | 60 |
| classification_coverage | 0.9851 | 0.9915 | 0.9989 | 0.9998 | 0.9994 | 1.0000 | 1.0000 | 1.0000 | 60 |
| greek_match_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.9036 | 0.9178 | 0.9410 | 0.9915 | 0.9858 | 0.9967 | 0.9974 | 0.9978 | 60 |

### QQQ
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.0564 | -0.0553 | -0.0436 | -0.0050 | -0.0041 | 0.0309 | 0.0371 | 0.0444 | 60 |
| delta_imbalance_ratio | -0.0355 | -0.0341 | -0.0258 | -0.0059 | -0.0019 | 0.0298 | 0.0327 | 0.0349 | 60 |
| net_dollar_delta_imbalance | -5681099279.6081 | -5608900763.9392 | -4045051466.7692 | -846790806.2389 | -310278297.8687 | 4269359920.3080 | 4934532299.6147 | 5116243891.2423 | 60 |
| gross_premium | 734445095.0000 | 779525711.1600 | 821029586.5000 | 1276307327.5000 | 1351256901.2833 | 2112327441.3500 | 2367395041.3600 | 2446004928.0000 | 60 |
| atm_iv | 0.1662 | 0.1666 | 0.1722 | 0.2049 | 0.2158 | 0.2637 | 0.2790 | 0.2808 | 60 |
| put_skew_25d | 0.0312 | 0.0314 | 0.0320 | 0.0488 | 0.0496 | 0.0646 | 0.0666 | 0.0676 | 48 |
| zero_dte_share | 0.2963 | 0.3243 | 0.3617 | 0.4232 | 0.4250 | 0.4921 | 0.5146 | 0.5359 | 60 |
| classification_coverage | 0.9938 | 0.9964 | 0.9986 | 0.9997 | 0.9995 | 0.9999 | 0.9999 | 1.0000 | 60 |
| greek_match_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.7471 | 0.8454 | 0.9428 | 0.9928 | 0.9841 | 0.9975 | 0.9986 | 0.9988 | 60 |

### IWM
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.1328 | -0.1290 | -0.1232 | -0.0289 | -0.0254 | 0.0579 | 0.1075 | 0.1180 | 60 |
| delta_imbalance_ratio | -0.1074 | -0.1027 | -0.0727 | -0.0101 | -0.0096 | 0.0486 | 0.0696 | 0.0991 | 60 |
| net_dollar_delta_imbalance | -1358855688.2199 | -1105877678.8200 | -801456542.3918 | -95659916.3094 | -100010475.0949 | 401736627.1594 | 907616524.7040 | 1459625464.1749 | 60 |
| gross_premium | 61382362.0000 | 65154646.7700 | 79207815.3000 | 155553794.0000 | 168687276.7167 | 314964798.9500 | 378296121.2800 | 397057949.0000 | 60 |
| atm_iv | 0.1606 | 0.1611 | 0.1623 | 0.1841 | 0.1854 | 0.2091 | 0.2242 | 0.2246 | 60 |
| put_skew_25d | 0.0303 | 0.0307 | 0.0311 | 0.0520 | 0.0494 | 0.0637 | 0.0679 | 0.0683 | 60 |
| zero_dte_share | 0.0533 | 0.0585 | 0.0794 | 0.1907 | 0.1964 | 0.3364 | 0.3489 | 0.3501 | 60 |
| classification_coverage | 0.9989 | 0.9993 | 0.9998 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 60 |
| greek_match_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.9016 | 0.9303 | 0.9569 | 0.9847 | 0.9823 | 0.9948 | 0.9966 | 0.9966 | 60 |

### SMH
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.2178 | -0.2032 | -0.1794 | -0.0064 | 0.0011 | 0.1905 | 0.2923 | 0.3039 | 60 |
| delta_imbalance_ratio | -0.4219 | -0.3979 | -0.2829 | -0.0212 | -0.0165 | 0.2959 | 0.3420 | 0.3789 | 60 |
| net_dollar_delta_imbalance | -1974642632.2979 | -1505940934.9438 | -735939771.1873 | -24939273.8313 | -54305217.1964 | 556223129.1919 | 834621700.1580 | 865122415.3723 | 60 |
| gross_premium | 43882964.0000 | 46044131.6400 | 49823910.1000 | 119275299.5000 | 155975162.0333 | 362538284.5500 | 461144734.5800 | 501370662.0000 | 60 |
| atm_iv | 0.3109 | 0.3137 | 0.3167 | 0.4346 | 0.4512 | 0.5802 | 0.5936 | 0.5949 | 60 |
| put_skew_25d | 0.0150 | 0.0156 | 0.0232 | 0.0438 | 0.0513 | 0.0921 | 0.0988 | 0.0990 | 60 |
| zero_dte_share | 0.0000 | 0.0000 | 0.0000 | 0.0320 | 0.0416 | 0.1375 | 0.1664 | 0.1692 | 60 |
| classification_coverage | 0.9872 | 0.9939 | 0.9994 | 1.0000 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 60 |
| greek_match_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.6807 | 0.7395 | 0.9135 | 0.9773 | 0.9679 | 0.9981 | 0.9991 | 0.9996 | 60 |

### TLT
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.2220 | -0.2061 | -0.1661 | 0.0050 | 0.0074 | 0.2383 | 0.3443 | 0.4001 | 60 |
| delta_imbalance_ratio | -0.2391 | -0.2186 | -0.1904 | 0.0045 | 0.0069 | 0.2699 | 0.3607 | 0.4441 | 60 |
| net_dollar_delta_imbalance | -277956077.1040 | -272763616.2700 | -262516876.2321 | 4707784.8663 | 16939538.7245 | 349672975.7139 | 531552813.2443 | 547892402.8856 | 60 |
| gross_premium | 12309802.0000 | 12774484.8200 | 14194160.1000 | 39269084.0000 | 47743614.9667 | 113493996.8000 | 132047448.4600 | 141091680.0000 | 60 |
| atm_iv | 0.0857 | 0.0864 | 0.0884 | 0.1063 | 0.1057 | 0.1249 | 0.1295 | 0.1320 | 60 |
| put_skew_25d | 0.0039 | 0.0060 | 0.0145 | 0.0241 | 0.0247 | 0.0360 | 0.0444 | 0.0465 | 60 |
| zero_dte_share | 0.0000 | 0.0000 | 0.0000 | 0.0421 | 0.0600 | 0.1767 | 0.3646 | 0.4404 | 60 |
| classification_coverage | 0.9998 | 0.9998 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| greek_match_coverage | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.3750 | 0.3810 | 0.4083 | 0.9307 | 0.8599 | 0.9946 | 0.9963 | 0.9966 | 60 |

Flagged extreme net_trade_sentiment observations (kept in the dataset, not removed):
  - TLT 2026-09-09: 0.400 (median 0.005, robust z=5.27)
  - TLT 2026-08-20: 0.306 (median 0.005, robust z=4.01)

### GLD
| Metric | min | p1 | p5 | median | mean | p95 | p99 | max | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| net_trade_sentiment | -0.2836 | -0.2291 | -0.1369 | 0.0048 | 0.0023 | 0.1124 | 0.3133 | 0.4300 | 60 |
| delta_imbalance_ratio | -0.5289 | -0.3695 | -0.1986 | 0.0239 | 0.0063 | 0.1856 | 0.3569 | 0.4804 | 60 |
| net_dollar_delta_imbalance | -3606128239.1509 | -2191361561.0264 | -500117499.6102 | 15895850.7505 | -43959919.2262 | 261237946.4213 | 996025888.8357 | 1479645469.4755 | 60 |
| gross_premium | 18663757.0000 | 23010941.3100 | 27567795.5000 | 60371536.0000 | 97369251.6000 | 284492197.9500 | 557948484.3600 | 755982631.0000 | 60 |
| atm_iv | 0.2077 | 0.2078 | 0.2140 | 0.2326 | 0.2339 | 0.2565 | 0.2612 | 0.2658 | 60 |
| put_skew_25d | -0.0254 | -0.0240 | -0.0207 | -0.0057 | 0.0013 | 0.0295 | 0.0320 | 0.0326 | 60 |
| zero_dte_share | 0.0000 | 0.0000 | 0.0000 | 0.0961 | 0.1054 | 0.2884 | 0.3552 | 0.3915 | 60 |
| classification_coverage | 0.9987 | 0.9989 | 0.9996 | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 | 60 |
| greek_match_coverage | 0.2214 | 0.6808 | 1.0000 | 1.0000 | 0.9870 | 1.0000 | 1.0000 | 1.0000 | 60 |
| oi_match_coverage | 0.8860 | 0.9011 | 0.9194 | 0.9766 | 0.9716 | 0.9940 | 0.9960 | 0.9967 | 60 |

Flagged extreme net_trade_sentiment observations (kept in the dataset, not removed):
  - GLD 2026-09-02: 0.430 (median 0.005, robust z=6.45)
  - GLD 2026-09-17: -0.284 (median 0.005, robust z=4.37)

## Item 7: reconciliation

**SPY**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $218,709,488, worst abs diff $592,870,903, median rel diff 17.9%, worst rel diff 27.0% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**QQQ**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $284,396,047, worst abs diff $662,969,272, median rel diff 22.9%, worst rel diff 29.7% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**IWM**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $42,801,832, worst abs diff $203,311,282, median rel diff 26.9%, worst rel diff 51.2% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**SMH**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $55,887,615, worst abs diff $228,238,062, median rel diff 45.4%, worst rel diff 72.4% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**TLT**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $16,128,653, worst abs diff $99,562,393, median rel diff 45.9%, worst rel diff 82.2% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**GLD**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $31,222,381, worst abs diff $445,804,949, median rel diff 51.9%, worst rel diff 75.4% (n=60). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=60). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

**ALL**  
- aggression (call/put bought+sold) vs gross premium: median abs diff $56,766,283, worst abs diff $662,969,272, median rel diff 29.8%, worst rel diff 82.2% (n=360). *call/put bought/sold necessarily excludes inside-spread (fractional-aggressor) premium; a positive gap is expected, not a bug.*
- DTE buckets vs eligible premium: median abs diff $0, worst abs diff $0, median rel diff 0.0%, worst rel diff 0.0% (n=360). *DTE buckets partition eligible premium with no overlap; should reconcile to floating-point precision. A material gap here is a real bug.*

## Item 8: cross-ETF comparability

| Ticker | Median gross premium | Median net $ delta | Median trades |
|---|---:|---:|---:|
| SPY | $1,341,642,345 | $-280,861,402 | 1429500 |
| QQQ | $1,276,307,328 | $-846,790,806 | 1034594 |
| IWM | $155,553,794 | $-95,659,916 | 166348 |
| SMH | $119,275,300 | $-24,939,274 | 20008 |
| GLD | $60,371,536 | $15,895,851 | 40379 |
| TLT | $39,269,084 | $4,707,785 | 21308 |

**Verdict:** Median gross premium spans a 34x range across tickers (SPY=$1,341,642,345, QQQ=$1,276,307,328, IWM=$155,553,794, SMH=$119,275,300, GLD=$60,371,536, TLT=$39,269,084); raw dollar magnitudes are not directly comparable cross-sectionally. Use net_trade_sentiment / delta_imbalance_ratio (already normalized to [-1, 1] / scale-free) for cross-sectional ranking, not raw premium or dollar delta.
