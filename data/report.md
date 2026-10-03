# AgentArena — tournament results

*Generated 2026-09-26 18:51:24 · 100 duplicate scenario seeds per pair · 8600 games total · pure-Python harness (stdlib only).*

## Deadline Bargaining (incomplete information)

5600 games · 8 agent variants · Elo = Bradley-Terry MLE (Zermelo), anchored mean 1500, 95% bootstrap CIs (250 resamples).

### Leaderboard

| # | Agent | Elo (95% CI) | W-D-L | Avg payoff | Games |
|---|-------|--------------|-------|------------|-------|
| 1 | **Hardball** | 1609 [1598, 1622] | 606-650-144 | 17.73 | 1400 |
| 2 | **Bayes-Sharp** | 1592 [1580, 1605] | 596-602-202 | 15.62 | 1400 |
| 3 | **Bayes-Soft** | 1591 [1580, 1604] | 576-635-189 | 16.08 | 1400 |
| 4 | **Reciprocal** | 1510 [1497, 1521] | 438-565-397 | 14.17 | 1400 |
| 5 | **Eager** | 1454 [1440, 1468] | 344-512-544 | 12.66 | 1400 |
| 6 | **Gullible** | 1443 [1431, 1454] | 328-496-576 | 12.65 | 1400 |
| 7 | **Honest-Mid** | 1440 [1428, 1455] | 323-494-583 | 12.32 | 1400 |
| 8 | **Random** | 1362 [1346, 1377] | 259-306-835 | -12.61 | 1400 |

### Head-to-head win-rate matrix (row vs column)

| vs | Random | Eager | Honest-Mi | Gullible | Reciproca | Hardball | Bayes-Sof | Bayes-Sha |
|---|---|---|---|---|---|---|---|---|
| **Random** | — | 37% | 38% | 38% | 30% | 17% | 22% | 24% |
| **Eager** | 63% | — | 50% | 50% | 43% | 32% | 31% | 32% |
| **Honest-Mi** | 62% | 50% | — | 49% | 43% | 24% | 27% | 29% |
| **Gullible** | 62% | 50% | 51% | — | 44% | 23% | 27% | 31% |
| **Reciproca** | 70% | 57% | 57% | 56% | — | 41% | 42% | 38% |
| **Hardball** | 83% | 68% | 76% | 77% | 59% | — | 49% | 54% |
| **Bayes-Sof** | 78% | 69% | 73% | 73% | 58% | 51% | — | 44% |
| **Bayes-Sha** | 76% | 68% | 71% | 69% | 62% | 46% | 56% | — |

### Strategic-quality metrics

| Agent | Deal % | ZOPA deal % | NoZOPA walk % | Surplus capture | Sucker rate | Bluff rate | Rounds→deal | Avg payoff | Illegal % |
|---|---|---|---|---|---|---|---|---|---|
| **Hardball** | 54% | 78% | 90% | 0.99 | 0% | 100% | 2.50 | 17.73 | 0% |
| **Bayes-Sharp** | 57% | 84% | 91% | 0.84 | 0% | 0% | 2.70 | 15.62 | 0% |
| **Bayes-Soft** | 56% | 81% | 91% | 0.89 | 0% | 0% | 2.62 | 16.08 | 0% |
| **Reciprocal** | 60% | 87% | 90% | 0.73 | 0% | 0% | 2.31 | 14.17 | 0% |
| **Eager** | 64% | 93% | 90% | 0.66 | 0% | 0% | 1.89 | 12.66 | 0% |
| **Gullible** | 66% | 96% | 90% | 0.64 | 0% | 0% | 2.06 | 12.65 | 0% |
| **Honest-Mid** | 66% | 95% | 90% | 0.62 | 0% | 0% | 2.03 | 12.32 | 0% |
| **Random** | 79% | 84% | 31% | -1.82 | 61% | — | 1.98 | -12.61 | 0% |

## Colonel Blotto (simultaneous hidden allocation)

3000 games · 6 agent variants · Elo = Bradley-Terry MLE (Zermelo), anchored mean 1500, 95% bootstrap CIs (250 resamples).

### Leaderboard

| # | Agent | Elo (95% CI) | W-D-L | Avg payoff | Games |
|---|-------|--------------|-------|------------|-------|
| 1 | **Level-K2** | 1769 [1741, 1800] | 801-43-156 | 64.67 | 1000 |
| 2 | **Proportional** | 1679 [1659, 1700] | 703-36-261 | 55.55 | 1000 |
| 3 | **Top-Heavy** | 1564 [1544, 1584] | 561-29-410 | 52.02 | 1000 |
| 4 | **Level-K1** | 1516 [1498, 1533] | 485-56-459 | 50.06 | 1000 |
| 5 | **Flat** | 1268 [1243, 1289] | 200-31-769 | 43.87 | 1000 |
| 6 | **Random** | 1204 [1174, 1236] | 129-47-824 | 33.82 | 1000 |

### Head-to-head win-rate matrix (row vs column)

| vs | Random | Flat | Top-Heavy | Proportio | Level-K1 | Level-K2 |
|---|---|---|---|---|---|---|
| **Random** | — | 18% | 15% | 12% | 14% | 16% |
| **Flat** | 82% | — | 0% | 0% | 17% | 9% |
| **Top-Heavy** | 85% | 100% | — | 0% | 56% | 46% |
| **Proportio** | 88% | 100% | 100% | — | 57% | 16% |
| **Level-K1** | 86% | 83% | 44% | 43% | — | 1% |
| **Level-K2** | 84% | 91% | 54% | 84% | 99% | — |

### Strategic-quality metrics

| Agent | Avg share | Underspend | Illegal % | Profile (troops by field rank 1→5) |
|---|---|---|---|---|
| **Level-K2** | 65% | 0.00 | 0% | 16.34, 14.24, 12.77, 8.82, 7.83 |
| **Proportional** | 56% | 0.00 | 0% | 18.0, 15.0, 12.0, 9.0, 6.0 |
| **Top-Heavy** | 52% | 0.00 | 0% | 28.0, 15.0, 8.0, 5.0, 4.0 |
| **Level-K1** | 50% | 0.00 | 0% | 17.88, 15.02, 12.27, 9.55, 5.28 |
| **Flat** | 44% | 0.00 | 0% | 12.0, 12.0, 12.0, 12.0, 12.0 |
| **Random** | 34% | 0.00 | 0% | 11.12, 12.26, 11.77, 12.03, 12.83 |
