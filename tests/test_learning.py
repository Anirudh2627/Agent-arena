"""Tests for the learnable policy and the RWR self-play loop."""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.agents.learning import (THETA_BOUNDS, LearnedNegotiator,
                                   clamp_theta, sample_theta)
from arena.agents.scripted_bargaining import BARGAINING_AGENTS
from arena.games.bargaining import P_HI, P_LO, BargainingGame
from arena.training.selfplay import evaluate_theta, train
from arena.tournament.runner import play_game

EXTREMES = []
for anchor in (0.25, 0.95):
    for bluff in (0.0, 35.0):
        EXTREMES.append({"anchor": anchor, "accept_mult": 0.6 if bluff == 0 else 1.8,
                         "bluff": bluff, "walk_p": 0.03 if anchor < 0.5 else 0.30,
                         "margin_scale": 0.5 if bluff < 10 else 1.6})


class TestLearnedNegotiator(unittest.TestCase):
    def test_clamp(self):
        th = clamp_theta({"anchor": 9, "accept_mult": -3, "bluff": 1e6,
                          "walk_p": -1, "margin_scale": 0})
        for k, (lo, hi) in THETA_BOUNDS.items():
            self.assertGreaterEqual(th[k], lo)
            self.assertLessEqual(th[k], hi)

    def test_first_offer_math(self):
        a = LearnedNegotiator("x", anchor=0.8, accept_mult=1.0, bluff=0.0,
                              walk_p=0.1, margin_scale=1.0)
        a.reset("buyer", {"role": "buyer", "your_value": 100}, {"seed": 1})
        self.assertEqual(a.first_offer(), 100 - int(round(0.8 * 80)))
        a.reset("seller", {"role": "seller", "your_cost": 50}, {"seed": 1})
        self.assertEqual(a.first_offer(), 50 + int(round(0.8 * 90)))
        # IR clamp: offers never cross own reservation
        a.reset("buyer", {"role": "buyer", "your_value": 40}, {"seed": 1})
        self.assertLessEqual(a.first_offer(), 39)
        a.reset("seller", {"role": "seller", "your_cost": 120}, {"seed": 1})
        self.assertGreaterEqual(a.first_offer(), 121)

    def test_claim_kinds(self):
        a = LearnedNegotiator("x", anchor=0.5, accept_mult=1.0, bluff=20.0,
                              walk_p=0.1, margin_scale=1.0)
        a.reset("buyer", {"role": "buyer", "your_value": 100}, {"seed": 1})
        self.assertEqual(a.claim_reservation(), 80)
        self.assertEqual(a.claim_kind(), "bluff")
        b = LearnedNegotiator("y", anchor=0.5, accept_mult=1.0, bluff=0.0,
                              walk_p=0.1, margin_scale=1.0)
        b.reset("seller", {"role": "seller", "your_cost": 50}, {"seed": 1})
        self.assertEqual(b.claim_reservation(), 50)
        self.assertEqual(b.claim_kind(), "honest")

    def test_extreme_thetas_stay_legal(self):
        """No theta corner may produce an illegal move or a self-harming
        proposal; engine violation counters must stay empty."""
        import random
        rng = random.Random(3)
        for i, th in enumerate(EXTREMES):
            for seed in (300 + i, 320 + i):
                g = BargainingGame(seed)
                agents = {"buyer": LearnedNegotiator("learner", **th),
                          "seller": BARGAINING_AGENTS["hardball"]()}
                r = play_game(g, agents)
                self.assertEqual(r["illegal"], {}, f"theta {th} seed {seed}")
                for d in r["decisions"]:
                    if d["seat"] == "buyer" and d["kind"] == "propose":
                        self.assertLessEqual(d["params"]["price"], g.v,
                                             "learner proposed above own value")
                        self.assertGreaterEqual(d["params"]["price"], P_LO)
                self.assertTrue(g.is_terminal())

    def test_theta_stamp_in_replay(self):
        g = BargainingGame(31)
        th = dict(sample_theta(__import__("random").Random(5)))
        agents = {"buyer": LearnedNegotiator("learner", **th),
                  "seller": BARGAINING_AGENTS["honest_mid"]()}
        r = play_game(g, agents)
        d0 = next(d for d in r["decisions"] if d["agent"] == "learner")
        self.assertEqual(d0["belief"]["policy"], "learned-rwr")
        self.assertIn("theta", d0["belief"])
        self.assertIn("claims", d0["belief"])


class TestSelfPlayLoop(unittest.TestCase):
    def test_train_smoke_and_determinism(self):
        h1 = train(gens=2, pop=4, train_seeds=2, eval_seeds=2, seed=77,
                   anchors=["honest_mid", "random"], progress=False)
        h2 = train(gens=2, pop=4, train_seeds=2, eval_seeds=2, seed=77,
                   anchors=["honest_mid", "random"], progress=False)
        self.assertEqual([g["fitness_best"] for g in h1["gens"]],
                         [g["fitness_best"] for g in h2["gens"]])
        self.assertEqual([g["theta"] for g in h1["gens"]],
                         [g["theta"] for g in h2["gens"]])
        g = h1["gens"][-1]
        for k, (lo, hi) in THETA_BOUNDS.items():
            self.assertGreaterEqual(g["theta"][k], lo)
            self.assertLessEqual(g["theta"][k], hi)
        self.assertIn("learner", g["eval"]["elo"])
        self.assertEqual(g["eval"]["metrics"]["illegal_rate"], 0.0)
        # champion ratchet: recorded eval Elo must be non-decreasing
        elos = [gg["eval"]["elo"]["learner"] for gg in h1["gens"]]
        self.assertEqual(elos, sorted(elos))
        # final holdout exists and is structured like an eval
        self.assertIn("final", h1)
        self.assertIn("holdout", h1["final"])
        self.assertIn("learner", h1["final"]["holdout"]["elo"])

    def test_evaluate_theta_structure(self):
        th = sample_theta(__import__("random").Random(1))
        ev = evaluate_theta(th, ["eager"], [8000, 8001])
        self.assertIn("elo", ev)
        self.assertIn("per_anchor", ev)
        self.assertIsNotNone(ev["per_anchor"]["eager"])
        self.assertEqual(ev["n_games"], 2 * 2)   # 1 pair x 2 seeds x 2 swaps


if __name__ == "__main__":
    unittest.main(verbosity=2)
