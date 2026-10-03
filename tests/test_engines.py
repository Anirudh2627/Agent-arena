"""Engine tests: legality, payoffs, deadlines, information hiding."""
import json
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.games.bargaining import (DELTA, ROUNDS, BargainingGame)
from arena.games.base import Action
from arena.games.blotto import BUDGET, N_FIELDS, BlottoGame


class TestBargaining(unittest.TestCase):
    def test_payoff_math(self):
        g = BargainingGame(1)
        g.v, g.c = 100, 40
        g.step({"buyer": Action("propose", {"price": 70})})     # t=1
        g.step({"seller": Action("propose", {"price": 90})})    # t=2
        g.step({"buyer": Action("accept", {})})                 # accepts 90 @ t=2
        pay = g.payoffs()
        self.assertAlmostEqual(pay["buyer"], DELTA ** 1 * (100 - 90), places=3)
        self.assertAlmostEqual(pay["seller"], DELTA ** 1 * (90 - 40), places=3)

    def test_deadline_final_phase_only_accept_walk(self):
        g = BargainingGame(2)
        for i in range(ROUNDS):
            seat = "buyer" if i % 2 == 0 else "seller"
            act, err = g.validate(seat, Action("propose", {"price": 50 + i}))
            self.assertIsNotNone(act, err)
            g.step({seat: act})
        # 6 proposals made -> final acceptance phase
        actor = g.acting_seats()[0]
        self.assertEqual(actor, "buyer")
        hint = g.action_hint(actor)
        self.assertNotIn("propose", hint["kinds"])
        self.assertIn("accept", hint["kinds"])
        _, err = g.validate(actor, Action("propose", {"price": 55}))
        self.assertIsNotNone(err)

    def test_illegal_moves_rejected(self):
        g = BargainingGame(3)
        a, err = g.validate("buyer", Action("propose", {"price": 999}))
        self.assertIsNone(a)
        self.assertIn("outside", err)
        a, err = g.validate("seller", Action("propose", {"price": 50}))
        self.assertIsNone(a)   # not seller's turn first
        a, err = g.validate("buyer", Action("accept", {}))
        self.assertIsNone(a)   # nothing on the table
        a, err = g.validate("buyer", Action("teleport", {}))
        self.assertIsNone(a)

    def test_walk_ends_game_zero_payoffs(self):
        g = BargainingGame(4)
        g.step({"buyer": Action("propose", {"price": 10}, "lowball")})
        g.step({"seller": Action("walk", {}, "insulted")})
        self.assertTrue(g.is_terminal())
        self.assertEqual(g.payoffs(), {"buyer": 0.0, "seller": 0.0})
        self.assertEqual(g.truth()["walked_by"], "seller")

    def test_information_hiding(self):
        """public_view must not leak any private value; private views are seat-scoped."""
        for seed in range(20):
            g = BargainingGame(seed)
            pub = json.dumps(g.public_view())
            v, c = g.v, g.c
            other = g.private_view("seller")
            self.assertNotIn("your_value", other)
            self.assertNotIn(v, other.values())
            # the exact secret pair must not be reconstructible from the public view
            self.assertNotIn('"v":', pub)
            self.assertNotIn('"c":', pub)
            for e in g.public_view()["history"]:
                self.assertNotIn("v", e)
                self.assertNotIn("c", e)

    def test_fallback_always_legal(self):
        for seed in range(30):
            g = BargainingGame(seed)
            for _ in range(ROUNDS + 2):
                if g.is_terminal():
                    break
                seat = g.acting_seats()[0]
                fb = g.fallback_action(seat)
                norm, err = g.validate(seat, fb)
                self.assertIsNotNone(norm, f"seed {seed}: fallback illegal ({err})")
                g.step({seat: norm})


class TestBlotto(unittest.TestCase):
    def test_tie_split_and_zero_sum(self):
        g = BlottoGame(5)
        a = Action("allocate", {"troops": [12] * N_FIELDS})
        g.step({"p0": g.validate("p0", a)[0], "p1": g.validate("p1", a)[0]})
        pay = g.payoffs()
        self.assertAlmostEqual(pay["p0"] + pay["p1"], 100.0, places=6)
        self.assertAlmostEqual(pay["p0"], 50.0, places=6)

    def test_overspend_rejected(self):
        g = BlottoGame(6)
        act, err = g.validate("p0", Action("allocate", {"troops": [500, 0, 0, 0, 0]}))
        self.assertIsNone(act)
        self.assertIn("exceeds budget", err)
        act, note = g.validate("p0", Action("allocate", {"troops": [10, -5, 10, 10, 10]}))
        self.assertIsNotNone(act)          # negatives clamped, still legal
        self.assertEqual(act.params["troops"][1], 0)
        self.assertTrue(note)

    def test_wrong_shape_rejected(self):
        g = BlottoGame(7)
        act, err = g.validate("p0", Action("allocate", {"troops": [1, 2, 3]}))
        self.assertIsNone(act)
        act, err = g.validate("p0", Action("allocate", {"troops": "sixty"}))
        self.assertIsNone(act)

    def test_secrecy_until_reveal(self):
        g = BlottoGame(8)
        g.validate("p0", Action("allocate", {"troops": [30, 10, 10, 5, 5]}))
        pub = json.dumps(g.public_view())
        self.assertNotIn("30, 10", pub.replace(" ", ""))
        self.assertNotIn("allocations", json.loads(pub))


if __name__ == "__main__":
    unittest.main(verbosity=2)
