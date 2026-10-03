"""AgentArena CLI.

Examples
--------
Offline scripted tournament (no API key needed):
    python -m arena.cli tournament --game bargaining --seeds 40

Add LLM agents to the ladder (OpenAI-compatible endpoint):
    export OPENAI_API_KEY=sk-...
    python -m arena.cli tournament --game bargaining --seeds 20 \\
        --llm "gpt4omini:strategic|model=gpt-4o-mini|variant=strategic" \\
        --llm "gpt4omini:basic|model=gpt-4o-mini|variant=basic" \\
        --llm "llama3|model=llama-3.1-8b-instruct|base_url=http://localhost:8000/v1|variant=strategic"

Plumbing test with the clearly-labeled MockLLM (no network):
    python -m arena.cli tournament --game bargaining --seeds 5 --llm "mocky|model=mock|variant=basic"

Export reward-weighted self-play data:
    python -m arena.cli sft-export --game bargaining --seeds 20 --out data/sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.agents.llm_agent import make_llm_agent
from arena.agents.scripted_bargaining import BARGAINING_AGENTS, DISPLAY
from arena.agents.scripted_blotto import BLOTTO_AGENTS, BLOTTO_DISPLAY
from arena.analysis.metrics import (all_metrics, bargaining_agent_metrics,
                                    blotto_agent_metrics)
from arena.games.bargaining import BargainingGame
from arena.games.blotto import BlottoGame
from arena.training.sft_export import export_sft
from arena.tournament.elo import (bootstrap_elo, elo_ratings, game_outcomes,
                                  winrate_matrix, wdl_counts)
from arena.tournament.runner import run_tournament

GAME_CLASSES = {"bargaining": BargainingGame, "blotto": BlottoGame}
GAME_AGENTS = {"bargaining": (BARGAINING_AGENTS, DISPLAY),
               "blotto": (BLOTTO_AGENTS, BLOTTO_DISPLAY)}


def _build_factories(game: str, agents_arg: str | None, llm_specs: list[str]) -> dict:
    base, _ = GAME_AGENTS[game]
    if agents_arg:
        factories = {a.strip(): base[a.strip()] for a in agents_arg.split(",") if a.strip()}
    else:
        factories = dict(base)
    for spec in llm_specs:
        name = spec.split("|", 1)[0]
        # fresh client per game -> fresh belief carry-over per game
        factories[name] = (lambda s=spec: make_llm_agent(s, record_prompts=True))
    return factories


def cmd_tournament(args):
    game = args.game
    cls = GAME_CLASSES[game]
    base, display = GAME_AGENTS[game]
    display = dict(display)
    factories = _build_factories(game, args.agents, args.llm or [])
    for name in factories:
        display.setdefault(name, name)
    seeds = [args.seed0 + i for i in range(args.seeds)]
    print(f"[{game}] {len(factories)} agents × {len(seeds)} duplicate seeds × seat-swap "
          f"= {len(factories)*(len(factories)-1)//2*len(seeds)*2} games")
    results = run_tournament(cls, factories, seeds, progress_every=500)
    outcomes = game_outcomes(results)
    names = list(factories)
    elo = elo_ratings(names, outcomes)
    ci = bootstrap_elo(names, outcomes, iters=args.bootstrap)
    wdl = wdl_counts(names, outcomes)
    _, rate, cnt = winrate_matrix(names, outcomes)
    print("\nLeaderboard:")
    for a in sorted(names, key=lambda x: -elo[x]):
        lo, hi = ci[a]
        w, d, l = wdl[a]["w"], wdl[a]["d"], wdl[a]["l"]
        print(f"  {display[a]:<22} Elo {elo[a]:6.0f} [{lo:5.0f},{hi:5.0f}]  W-D-L {w}-{d}-{l}")
    if args.json_out:
        payload = {"game": game, "elo": elo,
                   "ci": {k: list(v) for k, v in ci.items()}, "wdl": wdl,
                   "matrix": {"names": names, "rate": rate, "count": cnt},
                   "results_sample": results[:50]}
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {args.json_out}")
    if args.metrics:
        fn = bargaining_agent_metrics if game == "bargaining" else blotto_agent_metrics
        print("\nMetrics:")
        for m in sorted((fn(results, n) for n in names), key=lambda m: -elo[m["agent"]]):
            keys = [k for k in m if isinstance(m[k], (int, float)) and k not in ("agent", "games")]
            print(f"  {display[m['agent']]:<18} " + "  ".join(f"{k}={m[k]}" for k in keys[:9]))
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f)
        print(f"wrote {args.save_results} ({len(results)} games)")


def cmd_sft(args):
    cls = GAME_CLASSES[args.game]
    factories = _build_factories(args.game, args.agents, args.llm or [])
    seeds = [args.seed0 + i for i in range(args.seeds)]
    results = run_tournament(cls, factories, seeds)
    info = export_sft(results, args.out, keep_quantile=args.quantile, beta=args.beta)
    print(f"SFT export: kept {info['records_kept']}/{info['records_total']} decisions -> {args.out}")


def cmd_selfplay(args):
    from arena.training.selfplay import train
    anchors = [a.strip() for a in args.anchors.split(",")] if args.anchors else None
    print(f"Self-play RWR: gens={args.gens} pop={args.pop} train_seeds={args.train_seeds} "
          f"eval_seeds={args.eval_seeds} anchors={anchors or 'default'}")
    history = train(gens=args.gens, pop=args.pop, train_seeds=args.train_seeds,
                    eval_seeds=args.eval_seeds, seed=args.seed, anchors=anchors)
    print("\nGeneration summary (champion on fixed eval yardstick):")
    for g in history["gens"]:
        ev = g["eval"]
        print(f"  gen {g['gen']:>2}: Elo {ev['elo']['learner']:6.1f} "
              f"(hardball {ev['elo']['hardball']:6.1f}) {'*new champ*' if g['adopted'] else '':<11}| "
              f"bluffθ {g['theta']['bluff']:5.1f} rate {ev['metrics'].get('bluff_rate')} | "
              f"capture {ev['metrics'].get('capture')} | payoff {ev['metrics'].get('avg_payoff')}")
    fin = history.get("final")
    if fin:
        ho = fin["holdout"]
        print(f"  HOLDOUT ({ho['n_games']} games on unseen seeds): "
              f"Elo {ho['elo']['learner']:.1f} vs hardball {ho['elo']['hardball']:.1f}, "
              f"bayes_sharp {ho['elo']['bayes_sharp']:.1f} | capture {ho['metrics'].get('capture')} "
              f"| bluff_rate {ho['metrics'].get('bluff_rate')}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(history, f, indent=1)
        print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(prog="arena", description="AgentArena CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tournament", help="run a round-robin tournament")
    t.add_argument("--game", choices=list(GAME_CLASSES), required=True)
    t.add_argument("--agents", default=None, help="comma-separated subset of scripted agents")
    t.add_argument("--llm", action="append", help="LLM agent spec (see module docstring)")
    t.add_argument("--seeds", type=int, default=25)
    t.add_argument("--seed0", type=int, default=1000)
    t.add_argument("--bootstrap", type=int, default=150)
    t.add_argument("--metrics", action="store_true")
    t.add_argument("--json-out", default=None)
    t.add_argument("--save-results", default=None)
    t.set_defaults(fn=cmd_tournament)

    s = sub.add_parser("sft-export", help="run games and export reward-weighted training data")
    s.add_argument("--game", choices=list(GAME_CLASSES), required=True)
    s.add_argument("--agents", default=None)
    s.add_argument("--llm", action="append")
    s.add_argument("--seeds", type=int, default=10)
    s.add_argument("--seed0", type=int, default=1000)
    s.add_argument("--quantile", type=float, default=0.75)
    s.add_argument("--beta", type=float, default=2.0)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sft)

    p = sub.add_parser("selfplay", help="reward-weighted self-play training of the learned negotiator")
    p.add_argument("--gens", type=int, default=10)
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--train-seeds", type=int, default=6)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--anchors", default=None, help="comma-separated anchor agents (default: hardball,bayes_sharp,honest_mid,eager)")
    p.add_argument("--out", default=None, help="write history JSON here")
    p.set_defaults(fn=cmd_selfplay)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
