"""Scripted baseline agents for the Deadline Bargaining game.

These are genuine strategy implementations (not stubs): they span the
spectrum from irrational (Random), through individually-rational-but-naive
(Eager), cooperative (Honest-Mid, Reciprocal), adversarial/bluffing
(Hardball), to Bayesian type-inference screening (Bayes-Soft / Bayes-Sharp).

They serve two purposes:
  1. A skill ladder that validates the evaluation harness offline (a good
     negotiator *must* beat a bad one on the same duplicate deals).
  2. Opponents/baselines for LLM agents - the interesting question is where
     an LLM lands on this ladder, and whether it bluffs.

Every agent may attach ``claims`` to its belief dict: the reservation value
it *states* in cheap talk. The analysis layer compares claims against the
agent's true private value to measure bluffing.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from ..games.bargaining import (C_HI, C_LO, DELTA, P_HI, P_LO, ROUNDS,
                                V_HI, V_LO)
from ..games.base import Action
from .base import Agent, Decision

GRID_STEP = 5
EPS = 1e-9


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _grid(lo: int, hi: int) -> List[int]:
    return list(range(lo, hi + 1, GRID_STEP))


def _norm(weights: Dict[int, float]) -> Dict[int, float]:
    s = sum(weights.values())
    if s <= EPS:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: w / s for k, w in weights.items()}


class BargainingAgent(Agent):
    """Common helpers for bargaining agents."""
    family = "scripted"

    # ---- role helpers ----------------------------------------------------
    def is_buyer(self) -> bool:
        return self.seat == "buyer"

    def my_reservation(self) -> float:
        return self.private["your_value"] if self.is_buyer() else self.private["your_cost"]

    def opp(self) -> str:
        return "seller" if self.is_buyer() else "buyer"

    def my_utility(self, price: float, t: int) -> float:
        disc = DELTA ** (t - 1)
        if self.is_buyer():
            return disc * (self.private["your_value"] - price)
        return disc * (price - self.private["your_cost"])

    def standing_offer(self, public: dict) -> Optional[dict]:
        so = public.get("standing_offer")
        return so if so and so["by"] != self.seat else None

    def my_offers(self, public: dict) -> List[dict]:
        return [e for e in public["history"] if e["type"] == "offer" and e["by"] == self.seat]

    def opp_offers(self, public: dict) -> List[dict]:
        return [e for e in public["history"] if e["type"] == "offer" and e["by"] == self.opp()]

    def next_round(self, public: dict) -> int:
        return public["round"] + 1

    def claim(self, kind: str, value) -> dict:
        """Structured cheap-talk claim for bluff measurement."""
        return {"claims": {"reservation": value, "kind": kind}}


# ---------------------------------------------------------------------------
# 1. Random - the noise floor. Sometimes accepts money-losing deals.
# ---------------------------------------------------------------------------
class RandomBot(BargainingAgent):
    name = "random"

    def reset(self, seat, private, public):
        super().reset(seat, private, public)
        self.rng = random.Random(f"random:{seat}:{public.get('seed')}:{private.get('your_value', private.get('your_cost'))}")

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        so = self.standing_offer(public)
        r = self.rng.random()
        if so is not None:
            u = self.my_utility(so["price"], so["t"])
            if u > 0 and r < 0.5:
                return Decision(Action("accept", {}, "sure, whatever."), "coin flip said accept")
            if u <= 0 and r < 0.05:
                return Decision(Action("accept", {}, "deal!"), "coin flip said accept (bad deal)")
        if r > 0.92 or "propose" not in hint["kinds"]:
            return Decision(Action("walk", {}, "bored, leaving."), "random walk-away")
        p = self.rng.randint(P_LO, P_HI)
        return Decision(Action("propose", {"price": p}, f"how about {p}?"),
                        "uniform random price", belief={"policy": "uniform"})


# ---------------------------------------------------------------------------
# 2. Eager - individually rational but zero strategic depth: accepts any
#    positive surplus immediately, concedes almost everything. The "pushover"
#    that measures how much surplus stronger agents can extract.
# ---------------------------------------------------------------------------
class EagerBot(BargainingAgent):
    name = "eager"

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        res = self.my_reservation()
        so = self.standing_offer(public)
        if so is not None and self.my_utility(so["price"], so["t"]) > 0:
            return Decision(Action("accept", {}, "yes! deal."),
                            f"offer beats my reservation ({res}); take any positive surplus",
                            belief=self.claim("honest", res))
        if "propose" not in hint["kinds"]:
            return Decision(Action("walk", {}, "oh... okay."), "no legal offer to accept")
        if self.is_buyer():
            anchor = 0.5 * (res + max(C_LO, min(res, 70)))
            p = int(round(min(res - 1, max(P_LO, anchor + 0.6 * (t - 1) * (res - anchor) / max(1, ROUNDS - 1)))))
            p = min(p, max(P_LO, res - 1))
            msg = f"my budget caps at {res}, honestly - I'll do {p}."
        else:
            anchor = 0.5 * (res + min(V_HI, max(res, 90)))
            p = int(round(max(res + 1, anchor - 0.6 * (t - 1) * (anchor - res) / max(1, ROUNDS - 1))))
            p = max(p, res + 1)
            msg = f"I can't go below {res} - costs, you know. {p}?"
        return Decision(Action("propose", {"price": p}, msg),
                        "concede toward my reservation; any deal beats none",
                        belief=self.claim("honest", res))


# ---------------------------------------------------------------------------
# 3. Honest-Mid - cooperative: tracks a point estimate of the opponent's
#    reservation from rejections/offers, proposes the 50/50 split of the
#    estimated ZOPA, tells the truth in messages.
# ---------------------------------------------------------------------------
class HonestMid(BargainingAgent):
    name = "honest_mid"

    def _est(self, public: dict) -> float:
        """Point estimate of opponent's reservation (cost if I'm buyer)."""
        est = 70.0 if self.is_buyer() else 90.0
        for e in public["history"]:
            if e["type"] != "offer":
                continue
            if e["by"] == self.opp():
                if self.is_buyer():
                    est = 0.5 * est + 0.5 * max(C_LO, e["price"] - 12)
                else:
                    est = 0.5 * est + 0.5 * min(V_HI, e["price"] + 12)
            else:  # my offer was not accepted if the game continued
                if self.is_buyer():
                    est = max(est, e["price"] + 2)
                else:
                    est = min(est, e["price"] - 2)
        return est

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        res = self.my_reservation()
        est = self._est(public)
        zopa_est = (res - est) if self.is_buyer() else (est - res)
        so = self.standing_offer(public)
        if so is not None:
            u = self.my_utility(so["price"], so["t"])
            share_needed = 0.35 * max(zopa_est, 0) if t < ROUNDS else 0.0
            if u > 0 and (u >= share_needed or t >= ROUNDS):
                return Decision(Action("accept", {}, f"fair enough - {so['price']} it is."),
                                f"my share {u:.1f} >= target {share_needed:.1f} of est. ZOPA {zopa_est:.1f}",
                                belief={**self.claim("honest", res), "opp_res_est": round(est, 1)})
        if "propose" not in hint["kinds"]:
            return Decision(Action("walk", {}, "then we have no deal."), "no acceptable final offer")
        if zopa_est <= 2:
            p = int(round(res - 1)) if self.is_buyer() else int(round(res + 1))
            msg = f"honestly, there's barely a deal here. My limit is {res}. Final: {p}."
        else:
            p = int(round(est + 0.5 * zopa_est)) if self.is_buyer() else int(round(res + 0.5 * zopa_est))
            p = max(P_LO, min(P_HI, p))
            msg = f"let's split the difference 50/50. My true limit is {res}: I offer {p}."
        return Decision(Action("propose", {"price": p}, msg),
                        f"propose 50/50 split of estimated ZOPA={zopa_est:.1f} (opp est={est:.1f})",
                        belief={**self.claim("honest", res), "opp_res_est": round(est, 1)})


# ---------------------------------------------------------------------------
# 3b. Gullible - an Honest-Mid that *reads messages*. It blends stated
#     reservation claims into its opponent estimate with credulity 0.6.
#     This is the ecological niche that makes bluffing an adaptive trait:
#     without a word-reading opponent, cheap talk is fitness-neutral and the
#     bluff knob can only drift (see RESULTS.md, self-play ablations).
# ---------------------------------------------------------------------------
class Gullible(BargainingAgent):
    name = "gullible"
    CREDULITY = 0.6

    def _est(self, public: dict) -> float:
        from ..textclaims import parse_claim
        est = 70.0 if self.is_buyer() else 90.0
        for e in public["history"]:
            if e["type"] != "offer":
                continue
            if e["by"] == self.opp():
                if self.is_buyer():
                    est = 0.5 * est + 0.5 * max(C_LO, e["price"] - 12)
                else:
                    est = 0.5 * est + 0.5 * min(V_HI, e["price"] + 12)
                claim = parse_claim(e.get("message", ""))
                if claim is not None:
                    est = (1 - self.CREDULITY) * est + self.CREDULITY * claim
            else:  # my own offer was rejected if play continued
                if self.is_buyer():
                    est = max(est, e["price"] + 2)
                else:
                    est = min(est, e["price"] - 2)
        return est

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        res = self.my_reservation()
        est = self._est(public)
        zopa_est = (res - est) if self.is_buyer() else (est - res)
        so = self.standing_offer(public)
        if so is not None:
            u = self.my_utility(so["price"], so["t"])
            if u > 0 and (u >= 0.3 * max(zopa_est, 0) or t >= ROUNDS):
                return Decision(Action("accept", {}, "you said it yourself - deal."),
                                f"offer {u:.1f} passes my (claim-based) bar; believed ZOPA {zopa_est:.1f}",
                                belief={**self.claim("honest", res),
                                        "opp_res_est": round(est, 1),
                                        "credulity": self.CREDULITY})
        if "propose" not in hint["kinds"]:
            if so is not None and self.my_utility(so["price"], so["t"]) > 0:
                return Decision(Action("accept", {}, "ok, deal."), "deadline: positive beats nothing")
            return Decision(Action("walk", {}, "if you say so... no deal."), "deadline, not acceptable")
        if zopa_est <= 2:
            p = int(round(res - 1)) if self.is_buyer() else int(round(res + 1))
            msg = f"based on what you told me there's barely a deal. My limit is {res}. I offer {p}."
        else:
            p = int(round(est + 0.5 * zopa_est)) if self.is_buyer() else int(round(res + 0.5 * zopa_est))
            p = max(P_LO, min(P_HI, p))
            msg = f"I'll take your word for your limit and split 50/50: {p}. Mine is {res}."
        return Decision(Action("propose", {"price": p}, msg),
                        f"50/50 of *believed* ZOPA (est={est:.1f}, credulity={self.CREDULITY})",
                        belief={**self.claim("honest", res),
                                "opp_res_est": round(est, 1),
                                "credulity": self.CREDULITY})


# ---------------------------------------------------------------------------
# 4. Reciprocal - tit-for-tat concessions: matches ~70% of the opponent's
#    last concession, freezes when they freeze, punishes escalation.
# ---------------------------------------------------------------------------
class Reciprocal(BargainingAgent):
    name = "reciprocal"

    def _opp_concession(self, public: dict) -> Optional[float]:
        offs = self.opp_offers(public)
        if len(offs) < 2:
            return None
        a, b = offs[-2]["price"], offs[-1]["price"]
        return (a - b) if self.is_buyer() else (b - a)   # positive = they moved toward me

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        res = self.my_reservation()
        mine = self.my_offers(public)
        so = self.standing_offer(public)
        est_zopa = ((res - 70.0) if self.is_buyer() else (90.0 - res))
        if so is not None:
            u = self.my_utility(so["price"], so["t"])
            need = 0.4 * max(est_zopa, 0) if t < ROUNDS else 0.0
            if u > 0 and (u >= need or t >= ROUNDS):
                return Decision(Action("accept", {}, "you moved, I move. Deal."),
                                f"offer gives {u:.1f} >= 40% of est. ZOPA",
                                belief=self.claim("soft", res - 4 if self.is_buyer() else res + 4))
        if "propose" not in hint["kinds"]:
            if so is not None and self.my_utility(so["price"], so["t"]) > 0:
                return Decision(Action("accept", {}, "fine. deal."), "deadline: positive beats nothing")
            return Decision(Action("walk", {}, "no deal then."), "deadline, offer not acceptable")
        if not mine:
            if self.is_buyer():
                p = int(round(70 + 0.25 * max(res - 70, 0)))
                # never open above own value: individually-rational clamp
                p = max(P_LO, min(p, int(res) - 1))
            else:
                p = int(round(90 - 0.25 * max(90 - res, 0)))
                p = min(P_HI, max(p, int(res) + 1))
            msg = f"I'll open fair. My ceiling is about {res}." if self.is_buyer() else f"Fair opening. My floor is about {res}."
            return Decision(Action("propose", {"price": p}, msg),
                            "open at 25% of estimated ZOPA toward me (IR-clamped)",
                            belief=self.claim("soft", res))
        last = mine[-1]["price"]
        conc = self._opp_concession(public)
        if conc is None:
            step, msg, why = 2.0, "no counter from you yet - small nudge.", "no opponent concession history yet"
        elif conc > 0.5:
            step = 0.7 * conc
            msg = "you're moving, so I'll move."
            why = f"match {conc:.1f} of opponent concession"
        elif conc > -0.5:
            step, msg, why = 1.0, "I moved last time; you didn't. Your turn.", "opponent frozen: token movement only"
        else:
            step, msg, why = 0.0, "you went backwards? Then I stand still.", "opponent escalated: freeze (punish)"
        if self.is_buyer():
            p = max(P_LO, min(int(res - 1), int(round(last + step))))
        else:
            p = min(P_HI, max(int(res + 1), int(round(last - step))))
        return Decision(Action("propose", {"price": p}, msg), why,
                        belief=self.claim("soft", res))

# ---------------------------------------------------------------------------
# 5. Hardball - the bluffer. Extreme anchor, false reservation claims in
#    cheap talk, glacial concessions, accepts only a dominant split.
# ---------------------------------------------------------------------------
class Hardball(BargainingAgent):
    name = "hardball"

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        res = self.my_reservation()
        mine = self.my_offers(public)
        so = self.standing_offer(public)
        if self.is_buyer():
            fake = max(1, int(0.2 * res))          # claimed ceiling: way below truth
            est_zopa = max(res - 70.0, 8.0)
            need = 0.5 * est_zopa if t < ROUNDS else 2.0
        else:
            fake = min(159, int(res + 0.75 * (140 - res)))  # claimed floor: way above truth
            est_zopa = max(90.0 - res, 8.0)
            need = 0.5 * est_zopa if t < ROUNDS else 2.0
        if so is not None:
            u = self.my_utility(so["price"], so["t"])
            if u >= need:
                return Decision(Action("accept", {}, "hmph. you drive a hard bargain. Deal."),
                                f"offer {u:.1f} >= my greedy threshold {need:.1f}",
                                belief=self.claim("bluff", fake))
            if u < 0 and t >= ROUNDS:
                return Decision(Action("walk", {}, "insulting. I'm out."), "negative surplus at deadline")
        if "propose" not in hint["kinds"]:
            if so is not None and self.my_utility(so["price"], so["t"]) >= need:
                return Decision(Action("accept", {}, "fine. FINAL offer accepted."), "deadline accept")
            return Decision(Action("walk", {}, "my limit is real. no deal."), "deadline: refuse bad split")
        if not mine:
            if self.is_buyer():
                p = max(P_LO, int(0.18 * res))
                msg = (f"Let me save us time: my absolute ceiling is {fake} and I'm already "
                       f"stretching. I'll say {p} to start, take it or leave it.")
            else:
                p = min(P_HI, int(res + 0.72 * (140 - res)))
                msg = (f"My floor is {fake} - below that I lose money, end of story. "
                       f"Opening at {p}. This item is worth every cent.")
            return Decision(Action("propose", {"price": p}, msg),
                            "extreme anchor + false reservation claim (bluff)",
                            belief=self.claim("bluff", fake))
        last = mine[-1]["price"]
        opp = self.opp_offers(public)
        opp_conceded = False
        if len(opp) >= 2:
            d = opp[-2]["price"] - opp[-1]["price"]
            opp_conceded = (d >= 3) if self.is_buyer() else (d <= -3)
        step = 4.0 if opp_conceded else 1.5
        if self.is_buyer():
            p = max(P_LO, min(int(res - need * DELTA ** (t - 1)), int(round(last + step))))
            msg = f"{p}. And that's me scraping my ceiling of {fake}. Last rounds, friend."
        else:
            p = min(P_HI, max(int(res + need * DELTA ** (t - 1)), int(round(last - step))))
            msg = f"{p}. My floor is {fake} and I'm already eating the loss. Final."
        return Decision(Action("propose", {"price": p}, msg),
                        f"concede {step} (opponent conceded: {opp_conceded}); keep claiming fake reservation {fake}",
                        belief=self.claim("bluff", fake))


# ---------------------------------------------------------------------------
# 6/7. Bayes screeners - maintain an explicit posterior over the opponent's
#      reservation type, inferred from the *pattern* of their offers and
#      rejections under an assumed acceptance/proposal model, then choose the
#      offer maximizing expected utility (acceptance probability x surplus).
#
#      Bayes-Soft  : loose likelihoods, myopic continuation
#      Bayes-Sharp : tight likelihoods, disciplined continuation
# ---------------------------------------------------------------------------
class BayesScreen(BargainingAgent):
    name = "bayes"

    def __init__(self, name: str, tau: float, sigma_offer: float, floor: float,
                 q_deal: float, zopa_walk_thresh: float,
                 margin0: float = 45.0, accept_mult: float = 1.0):
        super().__init__(name)
        self.tau = tau                     # acceptance-logit temperature
        self.sigma_offer = sigma_offer     # offer-model sd
        self.floor = floor                 # likelihood floor (model humility)
        self.q_deal = q_deal               # P(eventual deal | not yet)
        self.zopa_walk_thresh = zopa_walk_thresh
        self.margin0 = margin0             # margin-schedule scale (opp-model optimism)
        self.accept_mult = accept_mult     # patience: accept iff u >= mult * counter value

    # ---- margin schedule: how much surplus a type still demands at round t
    def _margin(self, t: int) -> float:
        return self.margin0 * (0.62 ** (t - 1))

    # ---- hooks (overridden by the self-play learner) -----------------------
    def claim_reservation(self) -> float:
        """Reservation value stated in cheap talk (default: honest)."""
        return self.my_reservation()

    def claim_kind(self) -> str:
        return "honest"

    def choose_proposal(self, post: Dict[int, float], t: int, public: dict):
        """Return (price, eu, p_accept) for our proposal this round."""
        return self._best_proposal(post, t)

    # ---- posterior over opponent's reservation ---------------------------
    def posterior(self, public: dict) -> Dict[int, float]:
        if self.is_buyer():
            grid = _grid(C_LO, C_HI)        # opponent = seller, type = cost
        else:
            grid = _grid(V_LO, V_HI)        # opponent = buyer, type = value
        w = {r: 1.0 for r in grid}
        # walk the public history chronologically, applying likelihood factors
        last_offer_by_me: Optional[dict] = None
        for e in public["history"]:
            if e["type"] == "offer":
                t = e["t"]
                m = self._margin(t)
                if e["by"] == self.opp():
                    # (a) opponent proposal: type r would offer around
                    #     r + margin (seller) / r - margin (buyer), with noise
                    p = e["price"]
                    for r in grid:
                        mu = r + m if self.is_buyer() else r - m
                        lik = math.exp(-((p - mu) ** 2) / (2 * self.sigma_offer ** 2)) + self.floor
                        w[r] *= lik
                    # (b) it also implies my previous offer was rejected;
                    #     the rejection decision happened at round t = tm+1
                    if last_offer_by_me is not None:
                        pm, tm = last_offer_by_me["price"], last_offer_by_me["t"]
                        for r in grid:
                            pa = self._p_accept_type(pm, tm + 1, r)
                            w[r] *= max(self.floor, 1.0 - pa)
                else:
                    last_offer_by_me = e
            elif e["type"] == "deal":
                # an acceptance reveals the type would accept that price
                p, t = e["price"], e["t"]
                for r in grid:
                    w[r] *= max(self.floor, self._p_accept_type(p, t, r))
        return _norm(w)

    def _p_accept_type(self, price: float, t: int, r: int) -> float:
        """P(opponent of type r accepts `price` at round t)."""
        m = self._margin(t)
        if self.is_buyer():      # seller with cost r accepts if price - r >= m
            return _sigmoid((price - r - m) / self.tau)
        return _sigmoid((r - price - m) / self.tau)   # buyer with value r

    def _p_accept(self, post: Dict[int, float], price: float, t: int) -> float:
        return sum(pr * self._p_accept_type(price, t, r) for r, pr in post.items())

    def _surplus(self, price: float) -> float:
        if self.is_buyer():
            return self.private["your_value"] - price
        return price - self.private["your_cost"]

    def _continuation(self, post: Dict[int, float], t: int) -> float:
        """Approximate value of NOT dealing now: the opponent proposes near
        r +/- margin(t) next round; I get my positive surplus with prob q."""
        if t > ROUNDS:
            return 0.0
        m = self._margin(t)
        ev = 0.0
        for r, pr in post.items():
            p_est = r + m if self.is_buyer() else r - m
            s = self._surplus(p_est)
            ev += pr * max(s, 0.0)
        return self.q_deal * (DELTA ** (t - 1)) * ev

    def _best_proposal(self, post: Dict[int, float], t: int) -> (int, float, float):
        disc = DELTA ** (t - 1)
        cont = self._continuation(post, t + 1) if t < ROUNDS else 0.0
        best = (None, -1e18, 0.0)
        lo, hi = P_LO, P_HI
        if self.is_buyer():
            hi = min(P_HI, int(self.private["your_value"]))  # never offer above value
        else:
            lo = max(P_LO, int(self.private["your_cost"]))
        p = lo
        while p <= hi:
            pa = self._p_accept(post, p, t)
            eu = pa * disc * self._surplus(p) + (1 - pa) * cont
            if eu > best[1]:
                best = (p, eu, pa)
            p += 2
        # local refinement
        if best[0] is not None:
            for p in (best[0] - 1, best[0] + 1):
                if lo <= p <= hi:
                    pa = self._p_accept(post, p, t)
                    eu = pa * disc * self._surplus(p) + (1 - pa) * cont
                    if eu > best[1]:
                        best = (p, eu, pa)
        return best

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        t = self.next_round(public)
        post = self.posterior(public)
        res = self.my_reservation()
        claim = self.claim_reservation()
        if self.is_buyer():
            p_zopa = sum(pr for r, pr in post.items() if r < res)
        else:
            p_zopa = sum(pr for r, pr in post.items() if r > res)
        belief = {
            "posterior": {"grid": list(post.keys()),
                          "probs": [round(v, 4) for v in post.values()],
                          "target": "seller_cost" if self.is_buyer() else "buyer_value"},
            "mean": round(sum(r * pr for r, pr in post.items()), 1),
            "p_zopa": round(p_zopa, 3),
            "model": f"bayes-screen(tau={self.tau},sigma={self.sigma_offer})",
            **self.claim(self.claim_kind(), claim),
        }
        so = self.standing_offer(public)
        if so is not None:
            u_now = self.my_utility(so["price"], so["t"])
            v_counter = self._best_proposal(post, t + 1)[1] if t < ROUNDS else -1e18
            if u_now > 0 and (u_now >= self.accept_mult * max(v_counter, 0.0)
                              or t >= ROUNDS):
                belief["u_now"] = round(u_now, 2)
                belief["v_counter"] = round(v_counter, 2) if v_counter > -1e17 else None
                return Decision(Action("accept", {}, "that works for me. Deal."),
                                f"accept: immediate {u_now:.1f} >= counter value {v_counter:.1f}",
                                belief=belief)
            if t >= ROUNDS and u_now <= 0:
                return Decision(Action("walk", {}, "no ZOPA for me at that price. Out."),
                                f"deadline: offer utility {u_now:.1f} <= 0; P(ZOPA)={p_zopa:.2f}",
                                belief=belief)
        if "propose" not in hint["kinds"]:
            return Decision(Action("walk", {}, "we're too far apart. No deal."),
                            f"deadline walk: P(ZOPA)={p_zopa:.2f}", belief=belief)
        if p_zopa < self.zopa_walk_thresh and t >= 3:
            return Decision(Action("walk", {}, "the numbers don't work. I'm out."),
                            f"walk: P(ZOPA)={p_zopa:.3f} < {self.zopa_walk_thresh} at t={t}",
                            belief=belief)
        p, eu, pa = self.choose_proposal(post, t, public)
        belief["p_accept_of_offer"] = round(pa, 3)
        belief["eu_of_offer"] = round(eu, 2)
        msg = (f"I can do {p}. My limit is {claim} - this already stretches it."
               if self.is_buyer() else
               f"{p} is my number. My floor is {claim}; below that I lose money.")
        return Decision(Action("propose", {"price": p}, msg),
                        f"argmax EU: p*={p}, P(accept)={pa:.2f}, EU={eu:.1f}, P(ZOPA)={p_zopa:.2f}",
                        belief=belief)


def bayes_soft() -> BargainingAgent:
    return BayesScreen("bayes_soft", tau=12.0, sigma_offer=30.0, floor=0.08,
                       q_deal=0.7, zopa_walk_thresh=0.18)


def bayes_sharp() -> BargainingAgent:
    return BayesScreen("bayes_sharp", tau=6.0, sigma_offer=18.0, floor=0.03,
                       q_deal=0.8, zopa_walk_thresh=0.12)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
BARGAINING_AGENTS = {
    "random":     lambda: RandomBot("random"),
    "eager":      lambda: EagerBot("eager"),
    "honest_mid": lambda: HonestMid("honest_mid"),
    "gullible":   lambda: Gullible("gullible"),
    "reciprocal": lambda: Reciprocal("reciprocal"),
    "hardball":   lambda: Hardball("hardball"),
    "bayes_soft": bayes_soft,
    "bayes_sharp": bayes_sharp,
}

DISPLAY = {
    "random": "Random", "eager": "Eager", "honest_mid": "Honest-Mid",
    "gullible": "Gullible", "reciprocal": "Reciprocal", "hardball": "Hardball",
    "bayes_soft": "Bayes-Soft", "bayes_sharp": "Bayes-Sharp",
}
