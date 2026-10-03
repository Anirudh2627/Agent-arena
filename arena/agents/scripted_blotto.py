"""Scripted baseline agents for Colonel Blotto.

The ladder:
  random      - uniform random composition (noise floor)
  flat        - equal split (ignores field values)
  top_heavy   - concentrate on the richest fields (greedy heuristic)
  proportional- troops proportional to field values
  level_k1    - best response to a uniform-random opponent (sampled hill-climb)
  level_k2    - best response to the (noisy) level-k1 type distribution

Level-k responses are cached per shuffled value layout, so tournaments stay
fast. All randomness is seeded from the value layout -> deterministic.
"""
from __future__ import annotations

import random
from functools import lru_cache
from typing import Callable, List, Sequence, Tuple

from ..games.base import Action
from ..games.blotto import BUDGET, N_FIELDS
from .base import Agent, Decision


def _random_composition(rng: random.Random, budget: int, n: int) -> List[int]:
    if n == 1:
        return [budget]
    cuts = sorted(rng.sample(range(1, budget + n), n - 1)) if budget + n - 1 >= n - 1 else []
    # stars and bars via sorted cut points over budget+n-1 slots is fiddly;
    # simpler: sequential Dirichlet-like split
    remaining = budget
    out = []
    for i in range(n - 1):
        cap = remaining - (n - 1 - i)  # keep >=0 for the rest
        take = rng.randint(0, max(cap, 0)) if cap > 0 else 0
        out.append(take)
        remaining -= take
    out.append(remaining)
    return out


def _proportional(values: Sequence[int], budget: int = BUDGET) -> List[int]:
    tot = sum(values)
    alloc = [int(budget * v / tot) for v in values]
    i = 0
    while sum(alloc) < budget:
        alloc[i % N_FIELDS] += 1
        i += 1
    return alloc


def _expected_payoff(alloc: Sequence[int], samples: Sequence[Sequence[int]],
                     values: Sequence[int]) -> float:
    tot = 0.0
    for s in samples:
        for i, v in enumerate(values):
            if alloc[i] > s[i]:
                tot += v
            elif alloc[i] == s[i]:
                tot += v * 0.5
    return tot / max(1, len(samples))


def _hill_climb(sample_fn: Callable[[], Sequence[Sequence[int]]],
                values: Sequence[int], budget: int = BUDGET,
                max_passes: int = 40) -> List[int]:
    """Greedy 1-unit-move hill climb on expected payoff under `sample_fn`."""
    samples = sample_fn()
    alloc = _proportional(values, budget)
    cur = _expected_payoff(alloc, samples, values)
    for _ in range(max_passes):
        improved = False
        moves = [(i, j) for i in range(N_FIELDS) for j in range(N_FIELDS) if i != j]
        if sum(alloc) < budget:
            moves += [(-1, j) for j in range(N_FIELDS)]   # spend from reserve
        for i, j in moves:
            cand = list(alloc)
            if i >= 0:
                if cand[i] == 0:
                    continue
                cand[i] -= 1
            cand[j] += 1
            score = _expected_payoff(cand, samples, values)
            if score > cur + 1e-9:
                alloc, cur, improved = cand, score, True
        if not improved:
            break
    return alloc


@lru_cache(maxsize=4096)
def _uniform_samples(values_key: Tuple[int, ...], m: int = 260) -> Tuple[Tuple[int, ...], ...]:
    rng = random.Random("uniform:" + ",".join(map(str, values_key)))
    return tuple(tuple(_random_composition(rng, BUDGET, N_FIELDS)) for _ in range(m))


@lru_cache(maxsize=4096)
def level_k1_alloc(values_key: Tuple[int, ...]) -> Tuple[int, ...]:
    samples = _uniform_samples(values_key)
    return tuple(_hill_climb(lambda: samples, list(values_key)))


@lru_cache(maxsize=4096)
def _k1_noisy_samples(values_key: Tuple[int, ...], m: int = 90) -> Tuple[Tuple[int, ...], ...]:
    """Type distribution of a level-k1 player: its base allocation plus
    small perturbations (2-4 troops shuffled between fields)."""
    base = list(level_k1_alloc(values_key))
    rng = random.Random("k1noise:" + ",".join(map(str, values_key)))
    out = []
    for _ in range(m):
        a = list(base)
        for _ in range(rng.randint(1, 2)):
            i, j = rng.sample(range(N_FIELDS), 2)
            shift = rng.randint(1, 2)
            if a[i] >= shift:
                a[i] -= shift
                a[j] += shift
        out.append(tuple(a))
    return tuple(out)


@lru_cache(maxsize=4096)
def level_k2_alloc(values_key: Tuple[int, ...]) -> Tuple[int, ...]:
    samples = _k1_noisy_samples(values_key)
    return tuple(_hill_climb(lambda: samples, list(values_key)))


class BlottoAgent(Agent):
    family = "scripted"

    def _values(self, public: dict) -> List[int]:
        return list(public["field_values"])

    def submit(self, alloc: List[int], why: str, belief: dict | None = None) -> Decision:
        return Decision(Action("allocate", {"troops": list(alloc)}, ""),
                        why, belief or {})

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        raise NotImplementedError


class BlottoRandom(BlottoAgent):
    name = "blotto_random"

    def reset(self, seat, private, public):
        super().reset(seat, private, public)
        self.rng = random.Random(f"br:{seat}:{public.get('seed')}:{tuple(public.get('field_values', []))}")

    def act(self, public, private, hint) -> Decision:
        a = _random_composition(self.rng, BUDGET, N_FIELDS)
        return self.submit(a, "uniform random composition")


class BlottoFlat(BlottoAgent):
    name = "blotto_flat"

    def act(self, public, private, hint) -> Decision:
        base = BUDGET // N_FIELDS
        a = [base] * N_FIELDS
        a[0] += BUDGET - sum(a)
        return self.submit(a, "equal split across all fields")


class BlottoTopHeavy(BlottoAgent):
    name = "blotto_top_heavy"

    def act(self, public, private, hint) -> Decision:
        vals = self._values(public)
        order = sorted(range(N_FIELDS), key=lambda i: -vals[i])
        weights = [0.46, 0.24, 0.14, 0.09, 0.07]
        a = [0] * N_FIELDS
        for rank, idx in enumerate(order):
            a[idx] = int(BUDGET * weights[rank])
        i = 0
        while sum(a) < BUDGET:
            a[order[i % N_FIELDS]] += 1
            i += 1
        return self.submit(a, "concentrate on the highest-value fields",
                           {"profile": "top-heavy"})


class BlottoProportional(BlottoAgent):
    name = "blotto_proportional"

    def act(self, public, private, hint) -> Decision:
        a = _proportional(self._values(public))
        return self.submit(a, "troops proportional to field values")


class BlottoLevelK1(BlottoAgent):
    name = "blotto_levelk1"

    def act(self, public, private, hint) -> Decision:
        key = tuple(self._values(public))
        a = level_k1_alloc(key)
        return self.submit(list(a), "level-1: sampled best response to a uniform-random opponent",
                           {"level": 1})


class BlottoLevelK2(BlottoAgent):
    name = "blotto_levelk2"

    def act(self, public, private, hint) -> Decision:
        key = tuple(self._values(public))
        a = level_k2_alloc(key)
        return self.submit(list(a), "level-2: best response to the noisy level-1 type distribution",
                           {"level": 2})


BLOTTO_AGENTS = {
    "blotto_random":       lambda: BlottoRandom("blotto_random"),
    "blotto_flat":         lambda: BlottoFlat("blotto_flat"),
    "blotto_top_heavy":    lambda: BlottoTopHeavy("blotto_top_heavy"),
    "blotto_proportional": lambda: BlottoProportional("blotto_proportional"),
    "blotto_levelk1":      lambda: BlottoLevelK1("blotto_levelk1"),
    "blotto_levelk2":      lambda: BlottoLevelK2("blotto_levelk2"),
}

BLOTTO_DISPLAY = {
    "blotto_random": "Random", "blotto_flat": "Flat",
    "blotto_top_heavy": "Top-Heavy", "blotto_proportional": "Proportional",
    "blotto_levelk1": "Level-K1", "blotto_levelk2": "Level-K2",
}
