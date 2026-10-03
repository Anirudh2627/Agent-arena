# AgentArena — Results & Analysis

*Offline run · 100 duplicate scenario seeds per pair · seats swapped ·
8 bargaining agents (5,600 games) + 6 Blotto agents (3,000 games) +
self-play RWR training (~95,000 additional games across 3 population
conditions) · Elo = Bradley-Terry MLE anchored at mean 1500, 95 % bootstrap
CIs (250 resamples) · harness runtime ≈ 2 min total, pure stdlib Python.*

> All agents in this run are the scripted strategy ladder and the RWR-trained
> learned policy. The LLM harness plugs into the identical pipeline (see
> README); these results are the validated baseline an LLM must beat.

---

## 1. Deadline Bargaining — the strategy ladder

### Leaderboard

| # | Agent | Elo (95 % CI) | W-D-L | Avg payoff |
|---|-------|---------------|-------|------------|
| 1 | **Hardball** (bluffer) | **1609** [1598, 1622] | 606-650-144 | **17.73** |
| 2 | **Bayes-Sharp** (screener) | 1592 [1580, 1605] | 596-602-202 | 15.62 |
| 3 | **Bayes-Soft** (screener) | 1591 [1580, 1604] | 576-635-189 | 16.08 |
| 4 | Reciprocal (tit-for-tat) | 1510 [1497, 1521] | 438-565-397 | 14.17 |
| 5 | Eager (pushover) | 1454 [1440, 1468] | 344-512-544 | 12.66 |
| 6 | **Gullible** (reads cheap talk) | 1443 [1431, 1454] | 328-496-576 | 12.65 |
| 7 | Honest-Mid (cooperative) | 1440 [1428, 1455] | 323-494-583 | 12.32 |
| 8 | Random | 1362 [1346, 1377] | 259-306-835 | **−12.61** |

### Strategy metrics

| Agent | Deal % | ZOPA deal % (efficiency) | No-ZOPA walk % (discipline) | Capture | Sucker rate | Bluff rate | Opp. concession after bluff claims | Rounds→deal |
|---|---|---|---|---|---|---|---|---|
| Hardball | 54 % | 78 % | 90 % | **0.99** | 0 | **100 %** | **4.10** | 2.50 |
| Bayes-Sharp | 57 % | 84 % | **91 %** | 0.84 | 0 | 0 | — | 2.70 |
| Bayes-Soft | 56 % | 81 % | 91 % | 0.89 | 0 | 0 | — | 2.62 |
| Reciprocal | 60 % | 87 % | 90 % | 0.73 | 0 | 0 | — | 2.31 |
| Eager | 64 % | 93 % | 90 % | 0.66 | 0 | 0 | — | 1.89 |
| Gullible | 66 % | **96 %** | 90 % | 0.64 | 0 | 0 | — | 2.06 |
| Honest-Mid | 66 % | 95 % | 90 % | 0.62 | 0 | 0 | — | 2.03 |
| Random | 79 % | 84 % | **31 %** | **−1.82** | **61 %** | — | — | 1.98 |

35 % of the 100 seeded hands have no ZOPA (1,470 of 5,600 game instances);
pooled across the strategic ladder ~80 % of those correctly end in no deal.

### Key head-to-heads (row agent's score rate, 200 games each)

```
hardball    vs gullible    77.2 %     bayes_sharp vs gullible    69.2 %
hardball    vs honest_mid  75.5 %     bayes_sharp vs honest_mid  71.2 %
hardball    vs eager       68.2 %     bayes_sharp vs eager       67.5 %
hardball    vs reciprocal  59.2 %     bayes_sharp vs random      76.2 %
hardball    vs bayes_sharp 53.5 %     gullible    vs honest_mid  51.0 %
hardball    vs bayes_soft  48.8 %     reciprocal  vs honest_mid  56.8 %
```

### Findings

**F1 — Bluffing pays, and exactly whom it pays is measurable.**
Hardball lies in 100 % of its claims (stated ceiling ≈ 20 % of true value as
buyer; stated floor ≈ cost + ¾ of range as seller). Opponents concede **4.10**
price units on average after its turns vs 2.0–3.1 after cooperative agents'
honest claims, and Hardball extracts 77.2 % score rate from Gullible and
75.5 % from Honest-Mid. Against the Bayesian screeners it stalls at
53.5 % / 48.8 % — a statistical tie.

**F1b — Causal honesty about cheap talk.** Every scripted baseline *except
Gullible* is word-deaf: it infers from actions (offers/rejections) only. So
(i) screeners neutralize Hardball not by "seeing through lies" but by ignoring
words entirely and screening with its offer pattern; (ii) the 4.10-vs-2.x
concession gap is correlational — it tracks Hardball's whole posture (extreme
anchor + glacial concessions), with words as garnish. The *causal* effect of
false claims is isolated by the Gullible agent (Hardball's capture vs it:
0.729 vs 0.451 vs Bayes-Sharp — a controlled test, `tests/test_gullible.py`)
and by the self-play population ablation (§5, F9).

**F2 — Efficiency and distribution fight each other; Elo alone lies.**
Gullible closes 96 % of profitable hands (best efficiency) and ranks 6th;
Hardball closes 78 % but captures 0.99 of the pie. Any single metric
mis-ranks these agents; the vector + duplicate-deal Elo is what makes the
ordering defensible.

**F3 — Walking away is a measurable skill.** On no-ZOPA hands the rational
outcome is (0,0). The strategic ladder walks ~90 % of the time; Random
"deals" 69 % of the time, posts a 61 % sucker rate and −12.61 average payoff
— worse than doing nothing.

**F4 — Screeners pay for precision with speed.** Bayes-Sharp closes later
(2.70 rounds vs 1.89 for Eager): it screens with offers whose rejection
sharpens the posterior — and converts more surplus per deal than every
non-bluffer (capture 0.84).

**F5 — The harness caught a bug in its own baseline.** Reciprocal originally
posted a 16.3 % sucker rate: its opening offer anchored on the prior mean (70)
and could exceed its own value (buyer with v=43 opened at 70 — instant −27 if
accepted). Sucker-rate flagged it; the offer is now IR-clamped and a
regression test proves no strategic baseline proposes/accepts negative surplus
across 900 games. Cross-checked metrics working as intended.

---

## 2. Colonel Blotto

| # | Agent | Elo (95 % CI) | Avg share | Troops by field rank (1 = richest) | Rank-1 win % | Rank-5 win % |
|---|-------|---------------|-----------|------------------------------------|--------------|--------------|
| 1 | **Level-K2** | **1769** [1741, 1800] | **64.7 %** | 16.3 / 14.2 / 12.8 / 8.8 / 7.8 | 57.9 % | 62.4 % |
| 2 | Proportional | 1679 [1659, 1700] | 55.6 % | 18 / 15 / 12 / 9 / 6 | 57.4 % | 47.1 % |
| 3 | Top-Heavy | 1564 [1544, 1584] | 52.0 % | 28 / 15 / 8 / 5 / 4 | **97.6 %** | 17.4 % |
| 4 | Level-K1 | 1516 [1498, 1533] | 50.1 % | 17.9 / 15.0 / 12.3 / 9.6 / 5.3 | 49.6 % | 40.6 % |
| 5 | Flat | 1268 [1243, 1289] | 43.9 % | 12 / 12 / 12 / 12 / 12 | 19.6 % | **92.1 %** |
| 6 | Random | 1204 [1174, 1236] | 33.8 % | ≈ uniform | 23.3 % | 52.5 % |

**F6 — Level-k depth is only as good as the population model.** K2 beats K1
in **99 %** of games and Proportional 84 % — while K1, the "sophisticated"
best-responder to uniform-random play, *loses* to the one-line proportional
heuristic (**43 %**) because nobody in the population plays uniformly at
random. One step of *correct* meta-reasoning ≈ +250 Elo; one step of
mis-targeted meta-reasonation is worse than a robust heuristic.

**F7 — Concentration vs coverage is quantifiable.** Top-Heavy wins the
richest field 97.6 % of the time and still ends at 52 % share (forfeits the
tail); Flat mirrors the failure (92 % of the *cheapest* fields). K2's profile
holds every rank near 58–78 % win rate.

---

## 3. Self-play: does the agent *learn* to bluff?

**Setup.** `LearnedNegotiator` = the BayesScreen inference machinery with 5
bounded knobs: `anchor` (first-offer aggressiveness), `accept_mult`
(patience), `bluff` (cheap-talk offset — ≥10 counts as a measured bluff),
`walk_p` (P(ZOPA) walk threshold), `margin_scale` (opponent-model
hawkishness). Trainer: reward-weighted population evolution (RWR/RAFT-style)
— 20 particles, softmax fitness weighting, elites + perturbed children +
restarts. Methodology guards:

* **training hands change every generation** (no memorization of deals);
* **champion ratchet** on a fixed, deterministic evaluation yardstick
  (same 25 seeds every generation) → monotone, comparable Elo curve;
* **untouched holdout** (40 fresh seeds, 800 games) scored exactly once at
  the end — the honest generalization number;
* **identical mutation stream across population conditions** (same rng seed,
  same pop schedule) → any behavioral divergence is caused by the *opponent
  population*, not by luck of proposals.

### Main condition — anchors: Hardball, Bayes-Sharp, Honest-Mid, Eager, Gullible

```
champion θ:  anchor 0.54 · accept_mult 1.76 · bluff 35.0 (bound) · walk_p 0.27 · margin_scale 1.28
yardstick:   Elo 1655 (adopted gen 1, never beaten in 13 further generations)
HOLDOUT:     Elo 1672  (hardball 1593 · bayes_sharp 1548 · eager 1412 · gullible 1387 · honest_mid 1388)
per-anchor holdout score: hardball 55 % · bayes_sharp 69.4 % · honest_mid 86.3 % · eager 80.6 % · gullible 86.3 %
holdout metrics: payoff 15.18 · discipline 100 % · sucker 0 % · illegal 0 % · bluff rate 100 % · rounds→deal 3.85
```

**F8 — Reward-weighted self-play beat every hand-designed anchor and
generalizes.** On games it never saw, the learned policy outranks Hardball by
+79 Elo and Bayes-Sharp by +124 *within the same mini-ladder* (ratings are
only comparable inside a ladder), winning 55 %/69 % of hands against them and
>80 % against every cooperative type — with perfect no-ZOPA discipline, zero
sucker deals and zero illegal moves. The learned style is a **patient
closer**: it demands 1.76× the estimated counter-offer value before
accepting (deals land at round ~3.9, letting δ squeeze the opponent), walks
early on weak ZOPA beliefs, models opponents as hawkish, and — with a
word-reader in the population — pins its bluff knob at the 35-point bound.

### Population ablation — same trainer, same mutation stream, different worlds

| Condition | Anchors | bluff θ (start → end) | Holdout bluff rate | Holdout payoff | Learned substitutes for bluffing |
|---|---|---|---|---|---|
| main | hardball, bayes_sharp, honest_mid, eager, **gullible** | 17.9 → **35.0** (bound, gen 1) | **100 %** | 15.18 | — |
| coop | honest_mid, eager, reciprocal, **gullible** | 17.9 → **29.4** | **100 %** | 18.48 | — |
| resistant | bayes_sharp, hardball (**word-deaf**) | 13.5 → 16.5 → **3.5** (dropped at gen 5) | **0 %** | 10.79 | accept_mult **1.8** (bound: maximal patience), walk_p **0.04** (keeps negotiating), margin_scale **1.6** (bound: hawkish opponent model) |

**F9 — Deception evolves if and only if someone listens.** With a
word-reader in the population, false claims are causally rewarded (Gullible
blends stated limits into its estimate at credulity 0.6) and the bluff knob
saturates its upper bound within a generation or two. Against word-deaf
adversaries the *same* trainer with the *same* mutation proposals lets bluff
θ decay below the bluff threshold (measured bluff rate → 0 %) and instead
maxes out patience and opponent-model hawkishness. Deception here is not a
fixed "personality" of the algorithm — it is an adaptation to the
information-ecology of the population. (And the resistant-condition learner
still beats bayes_sharp 65.6 % of holdout hands *without* lying.)

**F10 — What actually beats a screener is acceptance discipline, not lies.**
Against word-deaf Bayes-Sharp the bluff knob is fitness-neutral; the learner's
edge comes from its tuned accept/walk policy (accept_mult 1.76–1.8, early
walks on weak ZOPA, late closes as δ compounds). Honest limitation: the
champion ratchet plateaued at gen 1 — a 5-D policy space against a *fixed*
anchor set saturates fast. Next steps: co-evolving anchors (non-stationary
populations), a larger policy space, and running the identical loop with LLM
policies via the SFT-export/re-fine-tune path.

---

## 4. Why this evaluation design holds up

1. **Duplicate deals** remove scenario luck: every pair faces identical
   private values/layouts, seats swapped. Rating gaps are strategy gaps.
2. **Relative-outcome scoring** works in variable-sum games where there is no
   "correct answer" — only better/worse negotiation on the same hand.
3. **Bradley-Terry + bootstrap CIs** give honest uncertainty: Hardball vs
   Bayes-Sharp (53.5 %, overlapping CIs) is correctly a tie; K2 vs K1 (99 %)
   is correctly domination.
4. **Orthogonal metrics** (efficiency, discipline, capture, sucker rate, bluff
   rate/effect, speed, illegal-move rate) explain *why* ratings order the way
   they do — and caught a real baseline bug (F5).
5. **Selection hygiene for learning**: fresh training hands, fixed yardstick,
   untouched holdout, shared mutation stream across ablation conditions.
6. **Guardrails**: engine-side validation with counted violations, strict
   overspend rejection, untrusted cheap talk, schema-enforced LLM output with
   retry→safe-fallback, information-hiding tests. The learned policies never
   produced an illegal move (0 % across ~95k games).

## 5. What an LLM run would add (harness ready, needs API keys)

* **Model-family axis:** same prompt variant, different models → Elo gaps.
* **Prompting axis:** `basic` vs `strategic` vs `belief` variants of one
  model — F6/F9 predict explicit belief-state carry-over matters most against
  bluffers and in Blotto.
* **Does the model bluff?** Bluff rate/effect metrics work on free text via
  claim-extraction regex; Gullible gives lies a causal payoff; the replay
  viewer flags bluffs against ground truth.
* **Self-play loop:** `sft-export` emits reward-weighted JSONL from
  top-quantile trajectories (exact chat prompts when `record_prompts=True`);
  fine-tune, re-enter the ladder, track Elo drift per generation.
