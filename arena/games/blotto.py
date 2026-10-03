"""Colonel Blotto: simultaneous, hidden resource allocation.

Two commanders split a fixed budget of troops across N battlefields with
public but unequal values. Allocations are submitted *sealed and
simultaneously*; on each field the higher allocation captures its value
(ties split it). Constant-sum => clean head-to-head outcomes for Elo.

Strategic depth: no pure-strategy equilibrium - good play requires mixed
strategies / level-k reasoning about what the opponent will do, which is
exactly the "reason about the other agent's beliefs" property we want.

Scenario per seed: the field values are a fixed multiset, shuffled.
Both seats see the same public values; allocations are private until reveal.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .base import Action, Game

N_FIELDS = 5
BUDGET = 60
FIELD_VALUES = [30, 25, 20, 15, 10]     # shuffled per seed; total = 100


class BlottoGame(Game):
    name = "blotto"
    simultaneous = True

    def __init__(self, seed: int):
        super().__init__(seed)
        rng = random.Random(f"blotto:{seed}")
        self.values: List[int] = rng.sample(FIELD_VALUES, len(FIELD_VALUES))
        self.allocs: Dict[str, List[int]] = {}
        self.terminal = False
        self._clamped: Dict[str, bool] = {}

    # ---- topology -------------------------------------------------------
    def seats(self) -> List[str]:
        return ["p0", "p1"]

    def is_terminal(self) -> bool:
        return self.terminal

    def acting_seats(self) -> List[str]:
        return [] if self.terminal else ["p0", "p1"]

    # ---- views -----------------------------------------------------------
    def public_view(self) -> Dict[str, Any]:
        view: Dict[str, Any] = {
            "game": self.name,
            "n_fields": N_FIELDS,
            "field_values": list(self.values),
            "budget": BUDGET,
            "terminal": self.terminal,
            "to_act": self.acting_seats(),
        }
        if self.terminal:
            view["allocations"] = {s: list(a) for s, a in self.allocs.items()}
            view["field_results"] = self._field_results()
            view["payoffs"] = self.payoffs()
        return view

    def private_view(self, seat: str) -> Dict[str, Any]:
        pv: Dict[str, Any] = {"role": seat, "budget": BUDGET,
                              "field_values": list(self.values)}
        if seat in self.allocs:
            pv["your_allocation"] = list(self.allocs[seat])
        return pv

    def truth(self) -> Dict[str, Any]:
        return {"values": list(self.values),
                "allocations": {s: list(a) for s, a in self.allocs.items()},
                "field_results": self._field_results() if self.terminal else None,
                "payoffs": self.payoffs() if self.terminal else None}

    # ---- moves -------------------------------------------------------------
    def action_hint(self, seat: str) -> Dict[str, Any]:
        return {
            "kinds": ["allocate"],
            "allocate": {
                "schema": {"kind": "allocate",
                           "params": {"troops": "<list of 5 ints>"},
                           "message": "<optional note recorded in the log>"},
                "n_fields": N_FIELDS, "budget": BUDGET,
                "constraint": "non-negative integers, sum <= budget",
            },
        }

    def validate(self, seat: str, action: Action) -> Tuple[Optional[Action], str]:
        if self.terminal:
            return None, "game already terminal"
        if seat not in self.acting_seats():
            return None, f"not {seat}'s turn"
        if (action.kind or "").lower() != "allocate":
            return None, f"unknown action kind: {action.kind!r}"
        troops = action.params.get("troops")
        if not isinstance(troops, (list, tuple)) or len(troops) != N_FIELDS:
            return None, f"troops must be a list of {N_FIELDS} ints"
        try:
            t = [int(round(float(x))) for x in troops]
        except (TypeError, ValueError):
            return None, "troops must be numeric"
        clamped = False
        if any(x < 0 for x in t):
            t = [max(0, x) for x in t]
            clamped = True
        if sum(t) > BUDGET:
            # strict anti-exploitation: over-budget submissions are REJECTED
            # (the runner counts the violation and substitutes a safe legal
            # fallback); they are never silently honored or rescaled.
            return None, f"troops sum {sum(t)} exceeds budget {BUDGET}"
        self._clamped[seat] = clamped
        return Action("allocate", {"troops": t}, (action.message or "")[:280]), \
               ("negative troops clamped to 0" if clamped else "")

    def fallback_action(self, seat: str) -> Action:
        # proportional allocation is always legal
        total = sum(self.values)
        base = [int(BUDGET * v / total) for v in self.values]
        i = 0
        while sum(base) < BUDGET:
            base[i % N_FIELDS] += 1
            i += 1
        return Action("allocate", {"troops": base}, "(engine fallback)")

    def step(self, actions: Dict[str, Action]) -> List[Dict[str, Any]]:
        if set(actions) != {"p0", "p1"}:
            raise ValueError("blotto needs sealed actions from both seats")
        for seat in ("p0", "p1"):
            self.allocs[seat] = list(actions[seat].params["troops"])
        self.terminal = True
        ev = {"type": "reveal", "allocations": {s: list(a) for s, a in self.allocs.items()},
              "field_results": self._field_results(), "payoffs": self.payoffs()}
        return [ev]

    # ---- outcome -------------------------------------------------------------
    def _field_results(self) -> List[dict]:
        out = []
        for i, val in enumerate(self.values):
            a, b = self.allocs["p0"][i], self.allocs["p1"][i]
            if a > b:
                win, pa, pb = "p0", val, 0.0
            elif b > a:
                win, pa, pb = "p1", 0.0, val
            else:
                win, pa, pb = "tie", val / 2.0, val / 2.0
            out.append({"field": i, "value": val, "p0_troops": a,
                        "p1_troops": b, "winner": win,
                        "p0_payoff": pa, "p1_payoff": pb})
        return out

    def payoffs(self) -> Dict[str, float]:
        if not self.terminal:
            return {"p0": 0.0, "p1": 0.0}
        res = self._field_results()
        return {"p0": round(sum(r["p0_payoff"] for r in res), 4),
                "p1": round(sum(r["p1_payoff"] for r in res), 4)}
