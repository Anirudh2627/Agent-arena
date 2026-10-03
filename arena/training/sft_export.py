"""Self-play data export for reward-weighted fine-tuning (RWR/RAFT-style).

For every decision in every tournament game we emit a training record:
  * if the decision came from an LLMAgent run with record_prompts=True, the
    record contains the exact chat `messages` + chosen `completion` (ready
    for SFT-style reward-weighted fine-tuning of the same model);
  * otherwise (scripted agents / no prompt capture) it contains an
    arena-native record (public view + action) usable to train small
    local policy models.

Weighting (reward-weighted regression): w = exp((payoff - baseline) / beta),
clipped to [0, w_max], where baseline is the agent's mean payoff on that
game type. Optionally keep only the top-q payoff fraction of games.

This is the 'optionally go further' RL hook from the project brief: run
tournaments with an LLM agent, export top-quartile trajectories, fine-tune,
re-enter the tournament, watch Elo move. No GPU needed at harness level.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence


def export_sft(results: Sequence[dict], out_path: str,
               keep_quantile: float = 0.75, beta: float = 2.0,
               w_max: float = 8.0) -> dict:
    # baselines: mean payoff per (game, agent)
    sums: Dict[tuple, List[float]] = {}
    for r in results:
        for seat, agent in r["seats"].items():
            sums.setdefault((r["game"], agent), []).append(r["payoffs"][seat])
    baseline = {k: (sum(v) / len(v) if v else 0.0) for k, v in sums.items()}

    # keep threshold: per-(game,agent) payoff quantile
    thresh: Dict[tuple, float] = {}
    for k, v in sums.items():
        s = sorted(v)
        idx = min(len(s) - 1, int(keep_quantile * len(s)))
        thresh[k] = s[idx] if keep_quantile < 1.0 else -math.inf

    n_out = n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            for d in r["decisions"]:
                seat = d["seat"]
                agent = d["agent"]
                payoff = r["payoffs"][seat]
                n_total += 1
                key = (r["game"], agent)
                if payoff < thresh.get(key, -math.inf):
                    continue
                w = min(w_max, math.exp((payoff - baseline.get(key, 0.0)) / beta))
                sft = (d.get("belief") or {}).get("_sft")
                if sft:
                    rec = {"messages": sft["messages"], "completion": sft["completion"],
                           "weight": round(w, 4)}
                else:
                    rec = {"arena_native": True,
                           "game": r["game"], "seed": r["seed"], "seat": seat,
                           "agent": agent,
                           "action": {"kind": d["kind"], "params": d.get("params", {}),
                                      "message": d.get("message", "")},
                           "reasoning": d.get("reasoning", ""),
                           "payoff": payoff, "weight": round(w, 4)}
                rec["meta"] = {"game": r["game"], "seed": r["seed"], "agent": agent,
                               "payoff": payoff, "baseline": round(baseline.get(key, 0.0), 3)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1
    return {"records_total": n_total, "records_kept": n_out,
            "keep_quantile": keep_quantile, "beta": beta, "path": out_path}
