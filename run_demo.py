#!/usr/bin/env python3
"""AgentArena one-command demo.

  python run_demo.py                 # full offline tournament + demo build
  python run_demo.py --quick         # fewer seeds (smoke test)
  python run_demo.py --seeds 150 --replays 10

Outputs (in data/):
  demo.html      self-contained replay viewer + Elo leaderboard + metrics
  results.json   full machine-readable payload
  report.md      markdown leaderboard/matrix/metrics tables
  sft_bargaining.jsonl / sft_blotto.jsonl   self-play training export
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arena.agents.scripted_bargaining import BARGAINING_AGENTS, DISPLAY
from arena.agents.scripted_blotto import BLOTTO_AGENTS, BLOTTO_DISPLAY
from arena.demo.build_demo import build_payload, render_demo, render_report
from arena.games.bargaining import BargainingGame
from arena.games.blotto import BlottoGame
from arena.training.sft_export import export_sft
from arena.tournament.runner import run_tournament


def main():
    ap = argparse.ArgumentParser(description="Run AgentArena tournaments and build the demo.")
    ap.add_argument("--seeds", type=int, default=100, help="duplicate scenario seeds per pair")
    ap.add_argument("--quick", action="store_true", help="shortcut for --seeds 25")
    ap.add_argument("--games", default="bargaining,blotto")
    ap.add_argument("--replays", type=int, default=8, help="curated replays per game")
    ap.add_argument("--out", default="data")
    ap.add_argument("--no-sft", action="store_true")
    ap.add_argument("--selfplay", action="store_true",
                    help="also run the RWR self-play loop and add its tab to the demo")
    ap.add_argument("--sp-gens", type=int, default=10)
    ap.add_argument("--sp-pop", type=int, default=16)
    ap.add_argument("--sp-train-seeds", type=int, default=6)
    ap.add_argument("--sp-eval-seeds", type=int, default=20)
    ap.add_argument("--sp-seed", type=int, default=11)
    ap.add_argument("--sp-ablation-gens", type=int, default=12)
    ap.add_argument("--no-ablations", action="store_true",
                    help="skip the population-ablation self-play conditions")
    args = ap.parse_args()
    seeds_n = 25 if args.quick else args.seeds
    seeds = [1000 + i for i in range(seeds_n)]
    want = [g.strip() for g in args.games.split(",") if g.strip()]

    specs = []
    if "bargaining" in want:
        specs.append({"title": "Deadline Bargaining (incomplete information)",
                      "game": "bargaining", "cls": BargainingGame,
                      "factories": BARGAINING_AGENTS, "display": DISPLAY})
    if "blotto" in want:
        specs.append({"title": "Colonel Blotto (simultaneous hidden allocation)",
                      "game": "blotto", "cls": BlottoGame,
                      "factories": BLOTTO_AGENTS, "display": BLOTTO_DISPLAY})
    if not specs:
        raise SystemExit(f"unknown games: {want}")

    print(f"AgentArena — {len(seeds)} duplicate seeds × round-robin × seat swap")
    payload = build_payload(specs, seeds, replay_budget=args.replays)

    if args.selfplay:
        from arena.training.selfplay import train
        print(f"\nSelf-play (RWR): {args.sp_gens} generations × pop {args.sp_pop}")
        history = train(gens=args.sp_gens, pop=args.sp_pop,
                        train_seeds=args.sp_train_seeds, eval_seeds=args.sp_eval_seeds,
                        seed=args.sp_seed)
        if not args.no_ablations:
            # Same mutation stream (identical rng seed), different opponent
            # populations -> isolates whether bluffing is *selected* or drifts.
            conds = [
                {"id": "coop",
                 "title": "Cooperative + gullible population (words believed)",
                 "anchors": ["honest_mid", "eager", "reciprocal", "gullible"]},
                {"id": "resistant",
                 "title": "Bluff-resistant population (word-deaf adversaries)",
                 "anchors": ["bayes_sharp", "hardball"]},
            ]
            history["ablations"] = []
            for cond in conds:
                print(f"  ablation [{cond['id']}]: anchors={cond['anchors']}")
                ab = train(gens=args.sp_ablation_gens, pop=args.sp_pop,
                           train_seeds=args.sp_train_seeds, eval_seeds=args.sp_eval_seeds,
                           seed=args.sp_seed, anchors=cond["anchors"], progress=False)
                history["ablations"].append({"id": cond["id"], "title": cond["title"],
                                             "anchors": cond["anchors"], **ab})
                fth = ab["final"]["theta"]
                print(f"    final theta bluff={fth['bluff']:.1f} | holdout bluff_rate="
                      f"{ab['final']['holdout']['metrics'].get('bluff_rate')} | "
                      f"holdout Elo {ab['final']['holdout']['elo']['learner']:.0f}")
        payload["selfplay"] = history
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "selfplay.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=True, indent=1)

    os.makedirs(args.out, exist_ok=True)
    demo_path = os.path.join(args.out, "demo.html")
    json_path = os.path.join(args.out, "results.json")
    md_path = os.path.join(args.out, "report.md")
    render_demo(payload, demo_path)
    render_report(payload, md_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    if not args.no_sft:
        # re-run a small slice purely to demonstrate the SFT export path
        for spec in specs:
            sub = run_tournament(spec["cls"], spec["factories"], seeds[:10])
            info = export_sft(sub, os.path.join(args.out, f"sft_{spec['game']}.jsonl"))
            print(f"SFT export [{spec['game']}]: kept {info['records_kept']}/{info['records_total']} decisions")

    # console summary
    for gname, g in payload["games"].items():
        print(f"\n=== {g['title']} — {g['games_played']} games, {g['runtime_s']}s ===")
        order = sorted(g["agents"], key=lambda a: -g["elo"][a])
        for i, a in enumerate(order, 1):
            lo, hi = g["ci"][a]
            w, d, l = g["wdl"][a]["w"], g["wdl"][a]["d"], g["wdl"][a]["l"]
            print(f" {i}. {g['display'][a]:<14} Elo {g['elo'][a]:6.0f} [{lo:.0f},{hi:.0f}]"
                  f"  W-D-L {w}-{d}-{l}  avg payoff {g['avg_payoff'][a]:.2f}")
    print(f"\nwrote {demo_path}\nwrote {json_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
