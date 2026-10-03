"""Deadline Bargaining: a two-player incomplete-information negotiation game.

Scenario (drawn from the seed, identical for every agent pair -> duplicate deals):
  * A Buyer with private value ``v ~ U[V_LO, V_HI]`` wants to buy an item.
  * A Seller with private cost   ``c ~ U[C_LO, C_HI]`` owns the item.
  * ZOPA (zone of possible agreement) exists iff v > c. Roughly a third of
    scenarios have NO ZOPA: the *rational* outcome there is to walk away.
    Agents that close negative-surplus deals are measurably bad negotiators.

Rules:
  * Players alternate proposals for a price p (Buyer first), up to ROUNDS
    proposals. After the final proposal the other player may only accept or
    walk (deadline pressure).
  * On your turn you may: propose(price, message), accept the standing offer,
    or walk away (game over, both get 0).
  * A deal at price p on proposal #t pays:
        Buyer:  delta^(t-1) * (v - p)
        Seller: delta^(t-1) * (p - c)
    Payoffs can be negative - accepting a bad price is allowed and punished.
  * ``messages`` are cheap talk: they are recorded verbatim in the transcript
    but the engine never trusts them. Claiming a false reservation value
    (bluffing) is a legitimate, measurable strategy.

Head-to-head outcome for Elo: compare the two payoffs in the same game
(win / draw / loss), i.e. "who negotiated the better deal on the same hand".
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .base import Action, Game

V_LO, V_HI = 40, 140      # buyer value support
C_LO, C_HI = 20, 120      # seller cost support
P_LO, P_HI = 0, 160       # legal price range
DELTA = 0.9               # per-round discount
ROUNDS = 6                # total proposals before deadline
MAX_MSG = 280             # cheap-talk length cap (guard against prompt spam)


class BargainingGame(Game):
    name = "bargaining"
    simultaneous = False

    def __init__(self, seed: int):
        super().__init__(seed)
        rng = random.Random(f"bargaining:{seed}")
        self.v = rng.randint(V_LO, V_HI)          # buyer value (secret)
        self.c = rng.randint(C_LO, C_HI)          # seller cost (secret)
        self.round = 0                            # proposals made so far
        self.table: Optional[dict] = None         # standing offer {"price", "by", "t"}
        self.history: List[dict] = []             # public events
        self.deal: Optional[dict] = None          # {"price", "t"}
        self.walked_by: Optional[str] = None
        self.terminal = False

    # ---- topology -------------------------------------------------------
    def seats(self) -> List[str]:
        return ["buyer", "seller"]

    def is_terminal(self) -> bool:
        return self.terminal

    def acting_seats(self) -> List[str]:
        if self.terminal:
            return []
        if self.round >= ROUNDS:
            # deadline: only the non-proposer may accept or walk
            last_by = self.table["by"] if self.table else "buyer"
            return ["seller" if last_by == "buyer" else "buyer"]
        return ["buyer"] if self.round % 2 == 0 else ["seller"]

    # ---- views ------------------------------------------------------------
    def public_view(self) -> Dict[str, Any]:
        return {
            "game": self.name,
            "round": self.round,                       # proposals made
            "rounds_total": ROUNDS,
            "final_acceptance_phase": self.round >= ROUNDS,
            "to_act": self.acting_seats(),
            "delta": DELTA,
            "price_range": [P_LO, P_HI],
            "public_prior": {"buyer_value": [V_LO, V_HI], "seller_cost": [C_LO, C_HI]},
            "standing_offer": dict(self.table) if self.table else None,
            "history": [dict(e) for e in self.history],
            "terminal": self.terminal,
        }

    def private_view(self, seat: str) -> Dict[str, Any]:
        base = {"role": seat, "delta": DELTA, "rounds_total": ROUNDS}
        if seat == "buyer":
            base["your_value"] = self.v
        else:
            base["your_cost"] = self.c
        return base

    def truth(self) -> Dict[str, Any]:
        zopa = self.v - self.c
        return {"v": self.v, "c": self.c, "zopa": zopa, "has_zopa": zopa > 0,
                "deal": self.deal, "walked_by": self.walked_by,
                "payoffs": self.payoffs()}

    # ---- moves -------------------------------------------------------------
    def action_hint(self, seat: str) -> Dict[str, Any]:
        at_deadline = self.round >= ROUNDS
        kinds = ["walk"] if at_deadline else ["propose", "walk"]
        if self.table is not None and self.table["by"] != seat:
            kinds.append("accept")
        return {
            "kinds": kinds,
            "propose": {"price_min": P_LO, "price_max": P_HI,
                        "schema": {"kind": "propose", "params": {"price": "<int>"},
                                   "message": "<<=280 chars of cheap talk>"}},
            "your_secret": ("value" if seat == "buyer" else "cost"),
        }

    def validate(self, seat: str, action: Action) -> Tuple[Optional[Action], str]:
        if self.terminal:
            return None, "game already terminal"
        if seat not in self.acting_seats():
            return None, f"not {seat}'s turn"
        kind = (action.kind or "").lower()
        msg = (action.message or "")[:MAX_MSG]
        at_deadline = self.round >= ROUNDS
        if kind == "walk":
            return Action("walk", {}, msg), ""
        if kind == "accept":
            if self.table is None or self.table["by"] == seat:
                return None, "no opponent offer on the table to accept"
            return Action("accept", {"price": self.table["price"],
                                      "t": self.table["t"]}, msg), ""
        if kind == "propose":
            if at_deadline:
                return None, "deadline reached: only accept/walk allowed"
            p = action.params.get("price", None)
            try:
                p = int(round(float(p)))
            except (TypeError, ValueError):
                return None, "propose requires a numeric price"
            if not (P_LO <= p <= P_HI):
                return None, f"price {p} outside [{P_LO},{P_HI}]"
            return Action("propose", {"price": p}, msg), ""
        return None, f"unknown action kind: {kind!r}"

    def fallback_action(self, seat: str) -> Action:
        """Safe legal default: accept if individually rational, else propose
        a mild midpoint offer, else walk (never an illegal move)."""
        at_deadline = self.round >= ROUNDS
        if self.table is not None and self.table["by"] != seat:
            p, t = self.table["price"], self.table["t"]
            disc = DELTA ** (t - 1)
            util = disc * ((self.v - p) if seat == "buyer" else (p - self.c))
            if util > 0:
                return Action("accept", {"price": p, "t": t}, "(engine fallback)")
        if not at_deadline:
            # propose own reservation nudged toward the middle of the prior ZOPA
            mid_prior = (V_LO + V_HI) / 2 - (C_LO + C_HI) / 2      # ~20
            if seat == "buyer":
                p = max(P_LO, min(P_HI, self.v - max(5, mid_prior)))
            else:
                p = min(P_HI, max(P_LO, self.c + max(5, mid_prior)))
            return Action("propose", {"price": int(p)}, "(engine fallback)")
        return Action("walk", {}, "(engine fallback)")

    def step(self, actions: Dict[str, Action]) -> List[Dict[str, Any]]:
        assert len(actions) == 1, "bargaining is strictly turn-based"
        seat, act = next(iter(actions.items()))
        events: List[dict] = []
        if act.kind == "propose":
            self.round += 1
            self.table = {"price": act.params["price"], "by": seat, "t": self.round}
            ev = {"type": "offer", "t": self.round, "by": seat,
                  "price": act.params["price"]}
            if act.message:
                ev["message"] = act.message
            self.history.append(ev)
            events.append(ev)
            if self.round >= ROUNDS:
                d = {"type": "deadline", "note": "final acceptance phase: accept or walk"}
                self.history.append(d)
                events.append(d)
        elif act.kind == "accept":
            p, t = self.table["price"], self.table["t"]
            self.deal = {"price": p, "t": t, "accepted_by": seat}
            self.terminal = True
            disc = DELTA ** (t - 1)
            ev = {"type": "deal", "t": t, "price": p, "accepted_by": seat,
                  "payoffs": self.payoffs(), "discount": round(disc, 4)}
            self.history.append(ev)
            events.append(ev)
        elif act.kind == "walk":
            self.walked_by = seat
            self.terminal = True
            ev = {"type": "walk", "by": seat, "payoffs": self.payoffs()}
            self.history.append(ev)
            events.append(ev)
        else:
            raise ValueError(f"illegal action reached step(): {act}")
        return events

    # ---- outcome -------------------------------------------------------------
    def payoffs(self) -> Dict[str, float]:
        if not self.deal:
            return {"buyer": 0.0, "seller": 0.0}
        p, t = self.deal["price"], self.deal["t"]
        disc = DELTA ** (t - 1)
        return {"buyer": round(disc * (self.v - p), 4),
                "seller": round(disc * (p - self.c), 4)}
