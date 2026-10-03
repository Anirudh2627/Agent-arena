"""Self-play with a lightweight RL signal: reward-weighted population
evolution over the LearnedNegotiator policy space (the brief's "optionally go
further" item — no gradients, no GPU, pure stdlib).

Loop per generation g:
  1. TRAIN   every particle theta_i plays the same seeded duplicate deals
             against a fixed anchor ladder (Hardball, Bayes-Sharp,
             Honest-Mid, Eager), seats swapped.
             fitness_i = mean payoff.
  2. WEIGHT  w_i = softmax((fitness_i - max) / beta_eff), beta_eff adapts to
             the population's fitness spread (reward-weighted regression /
             RAFT-style weighting).
  3. SELECT  elites survive verbatim; the rest of the next population is
             sampled proportional to w and Gaussian-perturbed (clamped to
             bounds); a small restart fraction keeps exploration alive.
  4. EVAL    the generation's best particle joins a fresh round-robin mini
             ladder against the anchors on FIXED evaluation seeds -> Elo +
             the full strategy-metric battery (bluff rate, capture,
             discipline...). Fixed eval seeds make the Elo-drift curve
             directly comparable across generations.

Everything is seeded: `train(seed=X)` is bit-for-bit reproducible. Training
seeds change each generation (so policies can't memorize hands); evaluation
seeds never change.
"""
from __future__ import annotations

import math
import random
import statistics
import time
from typing import Dict, List, Optional

from ..agents.learning import (THETA_BOUNDS, LearnedNegotiator, clamp_theta,
                               sample_theta)
from ..agents.scripted_bargaining import BARGAINING_AGENTS
from ..analysis.metrics import bargaining_agent_metrics
from ..games.bargaining import BargainingGame
from ..tournament.elo import elo_ratings, game_outcomes
from ..tournament.runner import play_game, run_tournament

LEARNER = "learner"
DEFAULT_ANCHORS = ["hardball", "bayes_sharp", "honest_mid", "eager", "gullible"]


# ------------------------------------------------------------------ training
def _fitness(theta: dict, anchors: List[str], seeds: List[int]) -> Dict[str, float]:
    """Mean payoff of `theta` across duplicate deals vs every anchor."""
    pays: List[float] = []
    per_anchor = {}
    for a in anchors:
        sub = []
        for seed in seeds:
            for flip in (False, True):
                game = BargainingGame(seed)
                if not flip:
                    agents = {"buyer": LearnedNegotiator(LEARNER, **theta),
                              "seller": BARGAINING_AGENTS[a]()}
                    seat = "buyer"
                else:
                    agents = {"buyer": BARGAINING_AGENTS[a](),
                              "seller": LearnedNegotiator(LEARNER, **theta)}
                    seat = "seller"
                r = play_game(game, agents)
                sub.append(r["payoffs"][seat])
        per_anchor[a] = round(statistics.fmean(sub), 3)
        pays.extend(sub)
    return {"mean": statistics.fmean(pays), "per_anchor": per_anchor, "n": len(pays)}


def _perturb(rng: random.Random, theta: dict, scale: float = 0.12) -> dict:
    out = {}
    for k, (lo, hi) in THETA_BOUNDS.items():
        sigma = scale * (hi - lo)
        if rng.random() < 0.10:                 # occasional bigger jump
            sigma *= 2.5
        out[k] = theta[k] + rng.gauss(0.0, sigma)
    return clamp_theta(out)


def _weighted_choice(rng: random.Random, population: List[dict],
                     weights: List[float]) -> dict:
    x = rng.random() * sum(weights)
    acc = 0.0
    for th, w in zip(population, weights):
        acc += w
        if x <= acc:
            return th
    return population[-1]


# ---------------------------------------------------------------- evaluation
def evaluate_theta(theta: dict, anchors: List[str], eval_seeds: List[int]) -> dict:
    """Mini round-robin: learner(best particle) + anchors on fixed seeds."""
    factories = {LEARNER: (lambda th=dict(theta): LearnedNegotiator(LEARNER, **th))}
    for a in anchors:
        factories[a] = BARGAINING_AGENTS[a]
    results = run_tournament(BargainingGame, factories, eval_seeds)
    outcomes = game_outcomes(results)
    names = list(factories)
    elo = elo_ratings(names, outcomes)
    met = bargaining_agent_metrics(results, LEARNER)
    per_anchor = {}
    for a in anchors:
        g = [(x, y, s) for x, y, s in outcomes if {x, y} == {LEARNER, a}]
        sc = sum(s if x == LEARNER else 1 - s for x, y, s in g)
        per_anchor[a] = round(sc / len(g), 3) if g else None
    slim_metrics = {k: met.get(k) for k in
                    ("avg_payoff", "deal_rate", "zopa_deal_rate", "nozopa_walk_rate",
                     "capture", "sucker_rate", "bluff_rate", "rounds_to_deal",
                     "conc_after_bluff", "conc_after_honest", "illegal_rate")}
    return {"elo": {k: round(v, 1) for k, v in elo.items()},
            "per_anchor": per_anchor, "metrics": slim_metrics,
            "n_games": len(results)}


# --------------------------------------------------------------------- loop
def train(gens: int = 8, pop: int = 16, train_seeds: int = 6, eval_seeds: int = 20,
          seed: int = 11, beta: float = 2.0, elite_frac: float = 0.3,
          restart_frac: float = 0.1, anchors: Optional[List[str]] = None,
          top_candidates: int = 2, holdout_seeds: int = 40,
          progress: bool = True) -> dict:
    """Reward-weighted population evolution with a champion ratchet.

    * training hands change every generation (no memorization of deals);
    * a fixed evaluation yardstick (same seeds every generation, deterministic
      agents) ranks challenger particles; the champion is only replaced when a
      challenger strictly beats it on that yardstick -> monotone Elo curve;
    * after training, the final champion is scored once on an untouched
      holdout seed set (the honest number to quote).
    """
    t0 = time.time()
    rng = random.Random(seed)
    anchors = list(anchors or DEFAULT_ANCHORS)
    eval_seed_list = [7000 + i for i in range(eval_seeds)]
    population = [sample_theta(rng) for _ in range(pop)]
    n_elite = max(2, int(round(elite_frac * pop)))
    n_restart = max(1, int(round(restart_frac * pop)))

    history = {"meta": {"gens": gens, "pop": pop, "train_seeds": train_seeds,
                        "eval_seeds": eval_seeds, "seed": seed, "beta": beta,
                        "anchors": anchors, "theta_bounds": THETA_BOUNDS,
                        "top_candidates": top_candidates,
                        "algorithm": "reward-weighted population evolution (RWR/RAFT-style) "
                                     "with champion ratchet on a fixed eval yardstick"},
               "gens": []}

    champ_theta: Optional[dict] = None
    champ_eval: Optional[dict] = None

    for gen in range(gens):
        gseeds = [5000 + gen * 100 + i for i in range(train_seeds)]
        fits = [_fitness(th, anchors, gseeds) for th in population]
        fvals = [f["mean"] for f in fits]
        fmax, fmean, fmin = max(fvals), statistics.fmean(fvals), min(fvals)
        spread = statistics.pstdev(fvals) if len(fvals) > 1 else 0.0
        beta_eff = max(beta, 0.5 * spread, 0.25)
        w = [math.exp((f - fmax) / beta_eff) for f in fvals]
        order = sorted(range(pop), key=lambda i: -fvals[i])

        # ---- challenge the champion with the generation's top candidates
        cand_idx, seen = [], set()
        for i in order:
            key = tuple(sorted(population[i].items()))
            if key not in seen:
                seen.add(key)
                cand_idx.append(i)
            if len(cand_idx) >= top_candidates:
                break
        candidates = [population[i] for i in cand_idx]
        cand_evals = [evaluate_theta(th, anchors, eval_seed_list) for th in candidates]
        best_c = max(range(len(candidates)),
                     key=lambda i: cand_evals[i]["elo"][LEARNER])
        adopted = False
        if champ_eval is None or \
                cand_evals[best_c]["elo"][LEARNER] > champ_eval["elo"][LEARNER]:
            champ_theta, champ_eval = candidates[best_c], cand_evals[best_c]
            adopted = True

        diversity = {k: round(statistics.pstdev([th[k] for th in population]), 4)
                     for k in THETA_BOUNDS}
        rec = {"gen": gen,
               "theta": {k: round(v, 3) for k, v in champ_theta.items()},
               "adopted": adopted,
               "candidates": [{"theta": {k: round(v, 3) for k, v in th.items()},
                               "train_fit": fvals[i],
                               "eval_elo": ev["elo"][LEARNER]}
                              for th, i, ev in zip(candidates, cand_idx, cand_evals)],
               "fitness_best": round(fmax, 2), "fitness_mean": round(fmean, 2),
               "fitness_worst": round(fmin, 2),
               "fitness_per_anchor": fits[order[0]]["per_anchor"],
               "diversity": diversity,
               "eval": champ_eval}
        history["gens"].append(rec)
        if progress:
            print(f"  gen {gen}: train fit(best/mean) {fmax:6.2f}/{fmean:6.2f} | "
                  f"CHAMP evalElo {champ_eval['elo'][LEARNER]:6.1f} "
                  f"(hardball {champ_eval['elo']['hardball']:6.1f}) "
                  f"{'*adopted*' if adopted else ''} "
                  f"theta[anchor={champ_theta['anchor']:.2f} bluff={champ_theta['bluff']:4.1f} "
                  f"acc={champ_theta['accept_mult']:.2f} walk={champ_theta['walk_p']:.2f} "
                  f"marg={champ_theta['margin_scale']:.2f}] "
                  f"bluff_rate={champ_eval['metrics'].get('bluff_rate')} "
                  f"capture={champ_eval['metrics'].get('capture')}", flush=True)

        if gen == gens - 1:
            break
        # ---- RWR update: elites + weighted-resampled perturbed children + restarts
        nxt = [dict(population[i]) for i in order[:n_elite]]
        while len(nxt) < pop - n_restart:
            nxt.append(_perturb(rng, _weighted_choice(rng, population, w)))
        while len(nxt) < pop:
            nxt.append(sample_theta(rng))
        population = nxt

    # ---- final honest number: untouched holdout, scored exactly once
    holdout = evaluate_theta(champ_theta, anchors,
                             [9000 + i for i in range(holdout_seeds)])
    history["final"] = {"theta": {k: round(v, 3) for k, v in champ_theta.items()},
                        "holdout": holdout}
    if progress:
        print(f"  holdout ({holdout_seeds} unseen seeds): Elo {holdout['elo'][LEARNER]:.1f} "
              f"(hardball {holdout['elo']['hardball']:.1f}, bayes_sharp "
              f"{holdout['elo']['bayes_sharp']:.1f}) | capture {holdout['metrics'].get('capture')} "
              f"| bluff_rate {holdout['metrics'].get('bluff_rate')}", flush=True)
    history["meta"]["runtime_s"] = round(time.time() - t0, 1)
    return history
