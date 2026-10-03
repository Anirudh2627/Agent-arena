"""The learnable negotiation policy used by the self-play loop.

``LearnedNegotiator`` is a BayesScreen (same posterior-inference machinery)
with five bounded, interpretable knobs — the search space of the RWR trainer:

  anchor        first-offer aggressiveness (fraction of the maximal demand
                between own reservation and the opponent's extreme prior)
  accept_mult   patience: accept iff immediate utility >= accept_mult x
                estimated counter-offer value (>1 = hold out, <1 = eager)
  bluff         cheap-talk offset: claimed reservation = true reservation
                minus (buyer) / plus (seller) this many points.
                bluff >= 10 is counted as a bluff by the metrics layer, so
                "does self-play learn to bluff?" is directly measurable.
  walk_p        P(ZOPA) collapse threshold for walking away
  margin_scale  opponent-model optimism: assumed surplus demand at round 1
                is 45 x margin_scale (shapes both inference and proposals)

Inference hyperparameters (tau, sigma, floor, q_deal) are fixed between the
soft/sharp baselines so evolution acts on *strategy*, not on likelihood
tuning. The policy stays inside the engine's legal space by construction
(first offers are IR-clamped), so learning cannot produce engine exploits.
"""
from __future__ import annotations

import random
from typing import Dict

from ..games.bargaining import C_LO, DELTA, P_HI, P_LO, V_HI
from ..games.base import Action
from .base import Decision
from .scripted_bargaining import BayesScreen

THETA_BOUNDS: Dict[str, tuple] = {
    "anchor":       (0.25, 0.95),
    "accept_mult":  (0.60, 1.80),
    "bluff":        (0.00, 35.0),
    "walk_p":       (0.03, 0.30),
    "margin_scale": (0.50, 1.60),
}

# inference hyperparameters, fixed across learning
FIXED = dict(tau=8.0, sigma_offer=22.0, floor=0.05, q_deal=0.78)


def sample_theta(rng: random.Random) -> Dict[str, float]:
    return {k: round(rng.uniform(lo, hi), 4) for k, (lo, hi) in THETA_BOUNDS.items()}


def clamp_theta(theta: Dict[str, float]) -> Dict[str, float]:
    return {k: min(hi, max(lo, float(theta[k]))) for k, (lo, hi) in THETA_BOUNDS.items()}


class LearnedNegotiator(BayesScreen):
    """Bayesian screener parameterized by theta (the RWR genome)."""
    family = "learning"

    def __init__(self, name: str, anchor: float, accept_mult: float, bluff: float,
                 walk_p: float, margin_scale: float):
        super().__init__(name, zopa_walk_thresh=walk_p,
                         margin0=45.0 * margin_scale, accept_mult=accept_mult,
                         **FIXED)
        self.theta = clamp_theta({"anchor": anchor, "accept_mult": accept_mult,
                                  "bluff": bluff, "walk_p": walk_p,
                                  "margin_scale": margin_scale})

    # ---- cheap talk: the learned bluff ------------------------------------
    def claim_reservation(self) -> float:
        res = self.my_reservation()
        b = self.theta["bluff"]
        return max(0, res - b) if self.is_buyer() else min(P_HI, res + b)

    def claim_kind(self) -> str:
        b = self.theta["bluff"]
        return "bluff" if b >= 10 else ("soft" if b >= 4 else "honest")

    # ---- first offer: the learned anchor -----------------------------------
    def first_offer(self) -> int:
        res = self.my_reservation()
        a = self.theta["anchor"]
        if self.is_buyer():
            p = res - a * max(res - C_LO, 1.0)
            return int(round(max(P_LO, min(res - 1, p))))
        p = res + a * max(V_HI - res, 1.0)
        return int(round(min(P_HI, max(res + 1, p))))

    def choose_proposal(self, post: Dict[int, float], t: int, public: dict):
        if not self.my_offers(public):
            p = self.first_offer()
            disc = DELTA ** (t - 1)
            pa = self._p_accept(post, p, t)
            cont = self._continuation(post, t + 1) if t < 6 else 0.0
            eu = pa * disc * self._surplus(p) + (1 - pa) * cont
            return p, eu, pa
        return self._best_proposal(post, t)

    # ---- wrap parent to stamp theta into the replay ------------------------
    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        dec = super().act(public, private, hint)
        first = "propose" in hint["kinds"] and not self.my_offers(public)
        dec.belief["policy"] = "learned-rwr"
        dec.belief["theta"] = {k: round(v, 3) for k, v in self.theta.items()}
        if first and dec.action.kind == "propose":
            dec.reasoning = (f"learned anchor {self.theta['anchor']:.2f} -> open at "
                             f"{dec.action.params['price']} (claiming {self.claim_reservation():.0f}); "
                             + dec.reasoning)
        return dec
