# AgentArena — multi-agent negotiation under conflicting incentives

LLM-ready agents that must **negotiate, bluff, and cooperate under incomplete
information** — plus a harness that measures *strategic quality* (not task
completion) across thousands of games, with Elo tournaments, a metrics battery,
and a step-through replay viewer that shows each agent's private reasoning and
beliefs next to the game state.

**Zero dependencies** (pure Python stdlib). Full offline pipeline runs in ~10 s:

```bash
python3 run_demo.py                        # tournaments + analysis + demo build (~15 s)
python3 run_demo.py --selfplay             # + RWR self-play loop & ablations (~2.5 min)
# open data/demo.html                      # replay viewer, Elo leaderboard, heatmap, self-play tab
python3 -m unittest discover tests         # 36 tests
node tests/smoke_demo.js data/demo.html    # optional headless UI test (needs jsdom)

python3 -m arena.cli selfplay --gens 14 --pop 20 --out data/selfplay.json
python3 -m arena.cli tournament --game bargaining --seeds 40 --metrics
```

> **Honesty note:** this repo ships and runs end-to-end offline using a
> *scripted strategy ladder* (Bayesian screeners, a bluffer, tit-for-tat, …).
> The LLM agent harness (`arena/agents/llm_agent.py`) is fully wired for any
> OpenAI-compatible endpoint — drop in `OPENAI_API_KEY` (or point
> `base_url` at Ollama/vLLM/OpenRouter) and LLM variants join the *same*
> tournament, produce the *same* replay/metric artifacts. A clearly-labeled
> `MockLLM` exercises that plumbing offline.

---

## The games

### 1. Deadline Bargaining (main game — negotiation, bluffing, screening)
Buyer with private value `v ~ U[40,140]`, Seller with private cost
`c ~ U[20,120]`. Alternating price offers (buyer first), pie shrinks
`δ = 0.9`/round, 6 proposals then a final accept-or-walk phase. Either side
may walk away any time (0, 0). A deal at price `p`, proposal `t` pays
`δ^{t-1}(v−p)` / `δ^{t-1}(p−c)` — **negative payoffs are allowed**, so
accepting a bad price is punished, not blocked. ~35 % of hands have **no
ZOPA**: the rational outcome is no deal, which makes *walking away* a
measurable skill. Every move can carry **cheap talk** the engine never trusts.

### 2. Colonel Blotto (simultaneous hidden allocation — level-k reasoning)
Sealed, simultaneous split of 60 troops over 5 fields worth
`[30,25,20,15,10]` (shuffled per seed). Higher allocation wins the field;
ties split. Constant-sum → clean head-to-head outcomes. No pure-strategy
equilibrium: good play requires modeling what the opponent will do.

## The agents

| Agent (bargaining) | Idea |
|---|---|
| `random` | noise floor; sometimes accepts money-losing deals |
| `eager` | individually rational pushover: accepts any positive surplus |
| `honest_mid` | point-estimate ZOPA tracking, always proposes 50/50, honest talk |
| `gullible` | honest 50/50 player that **reads cheap talk** — blends stated reservation claims into its estimate (credulity 0.6). The ecological niche that makes bluffing an *adaptive* trait instead of drift |
| `reciprocal` | tit-for-tat concessions: matches 70 % of your moves, freezes if you freeze |
| `hardball` | **bluffer**: extreme anchors, false reservation claims, glacial concessions, greedy accept threshold |
| `bayes_soft` / `bayes_sharp` | **belief-tracking screeners**: maintain an explicit posterior over the opponent's reservation type (updated from their offers/rejections under an acceptance model), then choose the offer maximizing `P(accept)·surplus + (1−P(accept))·continuation`; walk when `P(ZOPA)` collapses |
| `learner` (self-play) | the same BayesScreen machinery with **5 evolvable knobs** (anchor aggressiveness, patience, bluff offset, walk threshold, opponent-model hawkishness), trained by reward-weighted population evolution — see below |

Blotto ladder: `random`, `flat`, `top_heavy`, `proportional`, `level_k1`
(sampled best response to uniform-random play), `level_k2` (best response to
the noisy K1 type distribution). Level-k responses are cached per layout.

**LLM agents** (`arena/agents/llm_agent.py`): any OpenAI-compatible model ×
three prompt variants (`basic` / `strategic` / `belief` — the last one carries
a mandatory belief state the model must Bayesian-update every turn). Output is
strict JSON, extracted with a bracket matcher, *validated by the engine*,
retried once with the error message, else replaced by a safe fallback (and
counted). Opponent text is passed as **untrusted data** with explicit
anti-injection instructions.

```bash
export OPENAI_API_KEY=sk-...
python3 -m arena.cli tournament --game bargaining --seeds 20 --metrics \
  --llm "gpt4o-mini:strategic|model=gpt-4o-mini|variant=strategic" \
  --llm "gpt4o-mini:basic|model=gpt-4o-mini|variant=basic" \
  --llm "llama31|model=llama-3.1-8b-instruct|base_url=http://localhost:11434/v1|variant=strategic"
```

Cost: one LLM seat ≈ ≤ 4 calls/game ≈ 0.8 k tokens/call → a 20-seed duel vs
the full scripted ladder ≈ 280 games ≈ ~0.9 M tokens — small-model pocket
change, free-tier feasible.

## Evaluation methodology (the hard, differentiating part)

* **Duplicate deals.** Every pair plays the *same* seeded scenarios with seats
  swapped (21 pairs × 100 seeds × 2 = 4,200 bargaining games). Skill, not luck
  of the draw, drives rating gaps.
* **Relative-outcome Elo.** Each game is a W/D/L by payoff comparison in that
  game. Ratings = Bradley-Terry MLE (Zermelo iteration), anchored to mean
  1500, with 95 % bootstrap CIs (250 resamples).
* **Strategy-metric battery** (`arena/analysis/metrics.py`) — because "who won"
  is not "why": efficiency (deal rate on ZOPA hands), **discipline** (walk rate
  on no-ZOPA hands), **surplus capture**, **sucker rate** (negative-payoff
  deals), **bluff rate** (stated vs. true reservation) and **bluff effect**
  (opponent concession after bluff vs. honest claims), concession curves,
  rounds-to-deal, illegal-move rate.
* **Engine guardrails.** Every action passes `validate()`; illegal moves are
  replaced by a safe fallback and *counted per agent*; overspend is rejected
  outright; messages are length-capped; `private_view`/`public_view` form an
  information-hiding wall that tests actively probe (including an adversarial
  `IllegalAgent` and injection strings in cheap talk).
* **Self-play with a lightweight RL signal** (`arena/training/selfplay.py`):
  reward-weighted population evolution (RWR/RAFT-style) over the 5-knob
  learned policy — softmax fitness weighting, elites + perturbed children +
  restarts. Selection hygiene: training hands change every generation, a
  **champion ratchet** on a fixed deterministic yardstick keeps the Elo curve
  monotone and comparable, and an **untouched holdout** is scored exactly once
  at the end. Population ablations share one mutation stream, so behavioral
  divergence is attributable to the opponent population alone.
* **Self-play data for real fine-tuning.** `arena/training/sft_export.py`
  emits reward-weighted (RWR/RAFT-style) JSONL training records —
  top-quantile trajectories, weights `exp((payoff − baseline)/β)`; LLMAgent
  runs with `record_prompts=True` capture exact chat prompts for real
  fine-tuning loops.

## Demo viewer (`data/demo.html`, self-contained)

Tabs: **Leaderboard** (Elo bars + bootstrap CIs) · **Win-rate matrix**
(heatmap) · **Strategy metrics** (+ concession-curve small multiples, Blotto
rank profiles) · **Replays** — pick a curated game, step or auto-play through
it: offers, cheap talk, per-turn *internal reasoning*, live **belief
histograms** (posterior over the opponent's hidden type), claim chips that flip
to **✗ BLUFF** when you reveal ground truth, and the final payoff split ·
**Self-play** — Elo-drift curve, genome trajectories, behavior drift, the
population-ablation contrast, and the untouched-holdout report card.

## Results (offline ladder + self-play, 100 duplicate seeds)

See [`RESULTS.md`](RESULTS.md) for the full write-up. Headlines:

* The **bluffer tops the bargaining ladder** (Elo 1609) — opponents concede
  ~2× more after its turns (4.1 vs ~2–3 price units) — but Bayesian screeners
  **fully neutralize** it head-to-head (53.5 % / 48.8 %, within CIs): they
  weight actions over words.
* No single metric agrees with Elo: Gullible/Honest-Mid are the most
  *efficient* (96 %/95 % of ZOPA hands closed) yet capture the *least*;
  Hardball closes fewer deals but captures ~99 % of the pie.
* **Self-play works**: RWR over the 5-knob policy found a champion that, on an
  untouched holdout, outranks every hand-designed anchor in its mini-ladder
  (Elo 1672 vs Hardball 1593, Bayes-Sharp 1548) with perfect no-ZOPA
  discipline, zero sucker deals, zero illegal moves. Learned style: patient
  closer (accepts only at ≥1.76× counter value; deals land ~round 3.9).
* **Deception evolves iff someone listens** (controlled ablation, shared
  mutation stream): with the word-reading Gullible in the population the bluff
  knob saturates its bound (measured bluff rate 100 %); against word-deaf
  adversaries the *same trainer* lets bluffing decay to 0 % and instead maxes
  out patience and opponent-model hawkishness.
* In Blotto, **level-k reasoning pays only if the opponent model is right**:
  K2 dominates (64.7 % share, beats K1 99 %), but K1 *loses* to a one-line
  proportional heuristic (43 %) because it best-responds to a population
  nobody is playing.
* The metrics battery **caught a real bug in a baseline** during development
  (Reciprocal's unclamped opening offer → 16 % sucker rate), now covered by a
  regression test.

## Repository layout

```
agentarena/
├── run_demo.py                  # one-command offline pipeline (--selfplay adds the RWR loop)
├── arena/
│   ├── games/       base.py · bargaining.py · blotto.py     (engines, validation, info-hiding)
│   ├── agents/      base.py · scripted_bargaining.py · scripted_blotto.py · llm_agent.py
│   │                learning.py                             (5-knob policy evolved by self-play)
│   ├── tournament/  runner.py (duplicate-deal round robin) · elo.py (Bradley-Terry + bootstrap)
│   ├── analysis/    metrics.py (strategy battery) · report.py (markdown tables)
│   ├── replay/      export.py (replay JSON + interesting-game curation)
│   ├── training/    selfplay.py (RWR champion-ratchet loop + ablations) · sft_export.py
│   ├── textclaims.py            (shared cheap-talk claim parser: metrics + Gullible)
│   ├── demo/        template.html · build_demo.py  → data/demo.html
│   └── cli.py                   # tournament / sft-export / selfplay commands
├── tests/           test_engines.py · test_harness.py · test_learning.py · test_gullible.py
│                    smoke_demo.js                           (36 py tests + headless UI smoke)
└── data/            demo.html · results.json · report.md · selfplay.json · sft_*.jsonl
```

## Interview questions this repo answers concretely

* **“How do you evaluate an agent when there's no single right answer?”** —
  relative-outcome Elo on duplicate deals + an orthogonal metric vector
  (efficiency/discipline/capture/sucker/bluff) + behavioral regression tests.
  See `RESULTS.md` §“Why Elo alone lies”.
* **“How did you prevent agents from exploiting the engine?”** — validation on
  every move with counted violations and safe fallbacks, strict rejection of
  over-budget allocation, untrusted-data framing of cheap talk, schema-enforced
  JSON with retry→fallback, info-hiding tests, and an adversarial probe agent
  (`tests/test_harness.py::test_illegal_moves_never_honored`).
* **“What emergent behavior surprised you?”** — deception is
  *population-relative*: with the same mutation stream, the bluff knob
  saturates its bound when a word-reading opponent is present and decays to
  zero against word-deaf adversaries, which instead select for patience and
  hawkish opponent-modeling (RESULTS.md §3). Also: level-k depth backfires
  with a wrong population model, and the evaluation harness caught its own
  baseline's bug.

## Limitations & roadmap

* Scripted ladder + parametric learner validate the harness; the headline
  cross-model LLM results need API keys (the code path is tested with mocks
  and ready).
* The 5-D policy space saturates quickly against a *fixed* anchor set (the
  champion ratchet plateaued at gen 1) — next: co-evolving / non-stationary
  anchor populations and a larger policy space.
* Next game: 3-player coalition bargaining (majority vote → betrayal
  dynamics); then closing the full loop with LLM policies fine-tuned on the
  SFT export, re-entering the tournament per generation
