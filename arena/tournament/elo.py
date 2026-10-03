"""Elo-style ratings from head-to-head game outcomes.

Outcome definition per game: compare the two agents' payoffs *in the same
game* -> win (1) / draw (0.5) / loss (0). This yields a fair relative-strength
signal even in variable-sum games like bargaining (there is no single
"correct" answer, only better/worse negotiation on the same hand).

Rating model: Zermelo iteration (MLE of a Bradley-Terry model), anchored to
mean 1500, with 95% CIs from a game-level bootstrap. A tiny floor on total
score keeps winless agents finite.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

Triple = Tuple[str, str, float]    # (agentA, agentB, scoreA in [0,0.5,1])


def game_outcomes(results: Sequence[dict]) -> List[Triple]:
    out = []
    for r in results:
        seats = list(r["seats"].keys())
        s0, s1 = seats[0], seats[1]
        a, b = r["seats"][s0], r["seats"][s1]
        p0, p1 = r["payoffs"][s0], r["payoffs"][s1]
        score = 1.0 if p0 > p1 else (0.5 if p0 == p1 else 0.0)
        out.append((a, b, score))
    return out


def _zermelo(names: Sequence[str], triples: Sequence[Triple],
             iters: int = 300) -> Dict[str, float]:
    W = {n: 0.0 for n in names}
    N: Dict[Tuple[str, str], int] = {}
    for a, b, s in triples:
        W[a] += s
        W[b] += 1.0 - s
        N[(a, b)] = N.get((a, b), 0) + 1
        N[(b, a)] = N.get((b, a), 0) + 1
    gamma = {n: 1.0 for n in names}
    opps = {n: [m for m in names if m != n and (N.get((n, m), 0) > 0)] for n in names}
    for _ in range(iters):
        new = {}
        max_delta = 0.0
        for n in names:
            w = max(W[n], 0.5)                       # floor: keeps log finite
            denom = sum(N[(n, m)] / (gamma[n] + gamma[m]) for m in opps[n]) or 1e-9
            new[n] = w / denom
        # normalize geometric mean to 1
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        for n in names:
            new[n] /= gm
            max_delta = max(max_delta, abs(new[n] - gamma[n]))
        gamma = new
        if max_delta < 1e-10:
            break
    return gamma


def _to_elo(gamma: Dict[str, float]) -> Dict[str, float]:
    raw = {n: 400.0 * math.log10(max(g, 1e-12)) for n, g in gamma.items()}
    mean = sum(raw.values()) / len(raw)
    return {n: round(v - mean + 1500.0, 1) for n, v in raw.items()}


def elo_ratings(names: Sequence[str], triples: Sequence[Triple]) -> Dict[str, float]:
    return _to_elo(_zermelo(names, triples))


def bootstrap_elo(names: Sequence[str], triples: Sequence[Triple],
                  iters: int = 300, seed: int = 7,
                  ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
    rng = random.Random(seed)
    n = len(triples)
    samples = {name: [] for name in names}
    for _ in range(iters):
        boot = [triples[rng.randrange(n)] for _ in range(n)]
        r = _to_elo(_zermelo(names, boot, iters=120))
        for name in names:
            samples[name].append(r[name])
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    out = {}
    for name in names:
        s = sorted(samples[name])
        out[name] = (s[int(lo_q * len(s))], s[min(int(hi_q * len(s)), len(s) - 1)])
    return out


def winrate_matrix(names: Sequence[str], triples: Sequence[Triple]
                   ) -> Tuple[List[str], List[List[float]], List[List[int]]]:
    """rows[i][j] = score rate of names[i] against names[j] (0.5 diagonal)."""
    tot = {(a, b): [0.0, 0] for a in names for b in names if a != b}
    for a, b, s in triples:
        tot[(a, b)][0] += s
        tot[(a, b)][1] += 1
        tot[(b, a)][0] += 1 - s
        tot[(b, a)][1] += 1
    rate = [[0.5 if i == j else
             (tot[(names[i], names[j])][0] / tot[(names[i], names[j])][1]
              if tot[(names[i], names[j])][1] else 0.5)
             for j in range(len(names))] for i in range(len(names))]
    cnt = [[0 if i == j else tot[(names[i], names[j])][1]
            for j in range(len(names))] for i in range(len(names))]
    return list(names), rate, cnt


def wdl_counts(names: Sequence[str], triples: Sequence[Triple]) -> Dict[str, Dict[str, int]]:
    out = {n: {"w": 0, "d": 0, "l": 0} for n in names}
    for a, b, s in triples:
        if s > 0.5:
            out[a]["w"] += 1
            out[b]["l"] += 1
        elif s < 0.5:
            out[a]["l"] += 1
            out[b]["w"] += 1
        else:
            out[a]["d"] += 1
            out[b]["d"] += 1
    return out
