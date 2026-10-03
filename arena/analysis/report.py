"""Markdown/ASCII report rendering from the tournament payload."""
from __future__ import annotations

from typing import Dict, List


def _fmt(x, pct=False, nd=2):
    if x is None:
        return "—"
    if pct:
        return f"{100 * x:.0f}%"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def leaderboard_md(g: dict, display: Dict[str, str]) -> str:
    rows = sorted(g["agents"], key=lambda a: -g["elo"][a])
    lines = ["| # | Agent | Elo (95% CI) | W-D-L | Avg payoff | Games |",
             "|---|-------|--------------|-------|------------|-------|"]
    for i, a in enumerate(rows, 1):
        lo, hi = g["ci"][a]
        w, d, l = g["wdl"][a]["w"], g["wdl"][a]["d"], g["wdl"][a]["l"]
        ap = g["avg_payoff"].get(a)
        lines.append(f"| {i} | **{display.get(a, a)}** | {g['elo'][a]:.0f} "
                     f"[{lo:.0f}, {hi:.0f}] | {w}-{d}-{l} | {_fmt(ap)} | {g['games_per_agent']} |")
    return "\n".join(lines)


def matrix_md(g: dict, display: Dict[str, str]) -> str:
    names = g["matrix"]["names"]
    rate = g["matrix"]["rate"]
    head = "| vs | " + " | ".join(display.get(n, n)[:9] for n in names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [head, sep]
    for i, a in enumerate(names):
        cells = []
        for j in range(len(names)):
            cells.append("—" if i == j else f"{100 * rate[i][j]:.0f}%")
        lines.append(f"| **{display.get(a, a)[:9]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bargaining_metrics_md(g: dict, display: Dict[str, str]) -> str:
    cols = [("deal_rate", "Deal %", True), ("zopa_deal_rate", "ZOPA deal %", True),
            ("nozopa_walk_rate", "NoZOPA walk %", True), ("capture", "Surplus capture", False),
            ("sucker_rate", "Sucker rate", True), ("bluff_rate", "Bluff rate", True),
            ("rounds_to_deal", "Rounds→deal", False), ("avg_payoff", "Avg payoff", False),
            ("illegal_rate", "Illegal %", True)]
    head = "| Agent | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [head, sep]
    order = sorted(g["metrics"], key=lambda m: -g["elo"].get(m["agent"], 0))
    for m in order:
        cells = [_fmt(m.get(k), pct=p) for k, _, p in cols]
        lines.append(f"| **{display.get(m['agent'], m['agent'])}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def blotto_metrics_md(g: dict, display: Dict[str, str]) -> str:
    lines = ["| Agent | Avg share | Underspend | Illegal % | Profile (troops by field rank 1→5) |",
             "|---|---|---|---|---|"]
    order = sorted(g["metrics"], key=lambda m: -g["elo"].get(m["agent"], 0))
    for m in order:
        prof = m.get("rank_profile") or {}
        prof_s = ", ".join(str(prof.get(str(k), prof.get(k, "—"))) for k in range(1, 6))
        lines.append(f"| **{display.get(m['agent'], m['agent'])}** | {_fmt(m.get('avg_share'), pct=True)} "
                     f"| {_fmt(m.get('underspend'))} | {_fmt(m.get('illegal_rate'), pct=True)} | {prof_s} |")
    return "\n".join(lines)


def full_report(payload: dict) -> str:
    meta = payload["meta"]
    out = [f"# AgentArena — tournament results",
           f"",
           f"*Generated {meta['built_at']} · {meta['seeds']} duplicate scenario seeds per pair · "
           f"{meta['total_games']} games total · pure-Python harness (stdlib only).*",
           ""]
    for gname, g in payload["games"].items():
        display = g["display"]
        out += [f"## {g['title']}", "",
                f"{g['games_played']} games · {len(g['agents'])} agent variants · "
                f"Elo = Bradley-Terry MLE (Zermelo), anchored mean 1500, 95% bootstrap CIs "
                f"({meta['bootstrap_iters']} resamples).", "",
                "### Leaderboard", "", leaderboard_md(g, display), "",
                "### Head-to-head win-rate matrix (row vs column)", "", matrix_md(g, display), "",
                "### Strategic-quality metrics", ""]
        out += [(bargaining_metrics_md(g, display) if gname == "bargaining"
                 else blotto_metrics_md(g, display)), ""]
    return "\n".join(out)
