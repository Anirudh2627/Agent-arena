"""Builds the tournament payload (results.json) and the self-contained
demo viewer (demo.html) from a template + embedded JSON."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import time
from typing import Callable, Dict, List, Optional, Sequence

from ..agents.base import Agent
from ..analysis import metrics as M
from ..analysis.report import full_report
from ..replay.export import pick_replays, tag_game
from ..tournament.elo import (bootstrap_elo, elo_ratings, game_outcomes,
                              winrate_matrix, wdl_counts)
from ..tournament.runner import run_tournament

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template.html")


def build_game_payload(title: str, game_name: str, game_cls: Callable[[int], object],
                       factories: Dict[str, Callable[[], Agent]],
                       display: Dict[str, str], seeds: List[int],
                       replay_budget: int = 8, progress: bool = True) -> dict:
    t0 = time.time()
    results = run_tournament(game_cls, factories, seeds, progress_every=2000 if progress else 0)
    outcomes = game_outcomes(results)
    names = list(factories)
    elo = elo_ratings(names, outcomes)
    ci = bootstrap_elo(names, outcomes, iters=250)
    mnames, rate, cnt = winrate_matrix(names, outcomes)
    wdl = wdl_counts(names, outcomes)
    avg_payoff: Dict[str, float] = {}
    for n in names:
        ps = [r["payoffs"][s] for r in results for s, a in r["seats"].items() if a == n]
        avg_payoff[n] = round(sum(ps) / len(ps), 2) if ps else 0.0
    metric_fn = M.bargaining_agent_metrics if game_name == "bargaining" else M.blotto_agent_metrics
    mets = [metric_fn(results, n) for n in names]
    # replay curation: keep a diverse sample across both games
    replays = pick_replays(results, per_tag=2, max_total=replay_budget)
    for r in replays:
        r["display"] = {s: display.get(a, a) for s, a in r["seats"].items()}
    tagged_counts: Dict[str, int] = {}
    for r in results:
        for t in tag_game(r):
            tagged_counts[t] = tagged_counts.get(t, 0) + 1
    return {
        "title": title,
        "name": game_name,
        "agents": names,
        "display": display,
        "elo": elo,
        "ci": {k: [round(v[0], 1), round(v[1], 1)] for k, v in ci.items()},
        "wdl": wdl,
        "avg_payoff": avg_payoff,
        "matrix": {"names": mnames, "rate": [[round(x, 4) for x in row] for row in rate],
                   "count": cnt},
        "metrics": mets,
        "replays": replays,
        "games_played": len(results),
        "games_per_agent": 2 * (len(names) - 1) * len(seeds),
        "tag_counts": tagged_counts,
        "runtime_s": round(time.time() - t0, 1),
        "results_ref": results,     # popped before JSON serialization
    }


def build_payload(game_specs: Sequence[dict], seeds: List[int],
                  replay_budget: int = 8, progress: bool = True) -> dict:
    games = {}
    total = 0
    t0 = time.time()
    for spec in game_specs:
        pay = build_game_payload(spec["title"], spec["game"], spec["cls"],
                                 spec["factories"], spec["display"], seeds,
                                 replay_budget=replay_budget, progress=progress)
        total += pay["games_played"]
        pay.pop("results_ref")
        games[spec["game"]] = pay
    payload = {
        "meta": {
            "built_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seeds": len(seeds),
            "seed_range": [min(seeds), max(seeds)] if seeds else [],
            "total_games": total,
            "total_runtime_s": round(time.time() - t0, 1),
            "bootstrap_iters": 250,
            "python": platform.python_version(),
            "note": ("Offline run: all agents are scripted baselines. The LLM harness "
                     "(arena/agents/llm_agent.py) plugs OpenAI-compatible models into the "
                     "same pipeline; replays/leaderboards render identically."),
        },
        "games": games,
    }
    return payload


def render_demo(payload: dict, out_html: str) -> None:
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        tpl = f.read()
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    data = data.replace("</", "<\\/")     # safe inside <script>
    if "/*__DATA__*/null" not in tpl:
        raise RuntimeError("template.html missing the /*__DATA__*/null placeholder")
    html = tpl.replace("/*__DATA__*/null", data)
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)


def render_report(payload: dict, out_md: str) -> None:
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(full_report(payload))
