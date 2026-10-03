"""Replay export + interesting-game selection for the demo viewer."""
from __future__ import annotations

from typing import Dict, List, Sequence


def _slim_belief(belief: dict) -> dict:
    b = dict(belief or {})
    post = b.get("posterior")
    if isinstance(post, dict) and "probs" in post:
        b["posterior"] = {**post, "probs": [round(p, 4) for p in post["probs"]]}
    b.pop("_sft", None)
    return b


def to_replay(result: dict, tags: Sequence[str] = ()) -> dict:
    decs = []
    for d in result["decisions"]:
        decs.append({
            "seat": d["seat"], "agent": d["agent"], "step": d["step"],
            "kind": d["kind"], "params": d.get("params", {}),
            "message": d.get("message", ""),
            "reasoning": (d.get("reasoning") or "")[:900],
            "belief": _slim_belief(d.get("belief", {})),
        })
    truth = dict(result["truth"])
    truth.pop("payoffs", None)
    return {
        "id": f"{result['game']}-{result['seed']}-{'-'.join(sorted(set(result['seats'].values())))}",
        "game": result["game"],
        "seed": result["seed"],
        "seats": result["seats"],
        "tags": list(tags),
        "events": result["events"],
        "decisions": decs,
        "payoffs": result["payoffs"],
        "truth": truth,
        "illegal": result.get("illegal", {}),
    }


def tag_game(result: dict) -> List[str]:
    tags: List[str] = []
    t = result["truth"]
    agents = set(result["seats"].values())
    if result["game"] == "bargaining":
        has_zopa, deal = t["has_zopa"], t["deal"]
        if {"hardball"} & agents and any(a.startswith("bayes") for a in agents):
            tags.append("bayes_vs_hardball")
        if all(a.startswith("bayes") for a in agents):
            tags.append("bayes_duel")
        if deal:
            tags.append("deal")
            pb, ps = result["payoffs"]["buyer"], result["payoffs"]["seller"]
            if pb < 0 or ps < 0:
                tags.append("sucker_deal")
            if has_zopa and t["zopa"] > 0:
                share = max(pb, ps) / t["zopa"]
                if share >= 0.6:
                    winner = "buyer" if pb >= ps else "seller"
                    tags.append(f"dominant_{winner}")
            if deal["t"] >= 5:
                tags.append("deadline_deal")
        else:
            tags.append("no_deal")
            if not has_zopa:
                tags.append("nozopa_discipline")
        if not has_zopa:
            tags.append("no_zopa")
        for d in result["decisions"]:
            cl = (d.get("belief") or {}).get("claims") or {}
            if cl.get("kind") == "bluff":
                tags.append("bluff_used")
                break
    else:  # blotto
        tags.append("blotto")
        if any("levelk" in a for a in agents):
            tags.append("levelk2" if any("levelk2" in a for a in agents) else "levelk1")
        if all("levelk" in a for a in agents):
            tags.append("k_match")
        allocs = t.get("allocations") or {}
        for seat, alloc in allocs.items():
            if sum(alloc) < 60:
                tags.append("underspent")
            if alloc and max(alloc) >= 30:
                tags.append("concentrated")
    return tags


PRIORITY_TAGS = [
    "bayes_vs_hardball", "k_match", "bluff_used", "sucker_deal",
    "nozopa_discipline", "deadline_deal", "bayes_duel", "levelk2",
    "dominant_buyer", "dominant_seller", "concentrated", "no_deal",
    "underspent", "levelk1", "deal", "blotto",
]


def pick_replays(results: Sequence[dict], per_tag: int = 2,
                 max_total: int = 12, seed: int = 17) -> List[dict]:
    """Curate a diverse, interesting replay set: distinct matchups and seeds,
    prioritized by dramatic/analytically useful tags."""
    import random as _random
    rng = _random.Random(seed)
    tagged: List[tuple] = [(tag_game(r), r) for r in results]
    rng.shuffle(tagged)
    chosen: List[dict] = []
    used_seeds, used_pairs = set(), set()

    def pair_of(r):
        return tuple(sorted(set(r["seats"].values())))

    for tag in PRIORITY_TAGS:
        got = 0
        # within a tag, prefer games that actually closed a deal (more to watch)
        matches = sorted([( tags, r) for tags, r in tagged if tag in tags],
                         key=lambda tr: 0 if "deal" in tr[0] else 1)
        for tags, r in matches:
            if got >= per_tag or len(chosen) >= max_total:
                break
            if r["seed"] in used_seeds or pair_of(r) in used_pairs:
                continue
            chosen.append(to_replay(r, tags))
            used_seeds.add(r["seed"])
            used_pairs.add(pair_of(r))
            got += 1
    return chosen
