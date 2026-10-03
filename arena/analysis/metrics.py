"""Strategic-quality metrics - the 'how do you evaluate an agent when there
is no single right answer?' layer.

Elo answers *relative strength*. These metrics answer *why*:

Bargaining:
  deal_rate          - fraction of games that ended in a deal
  zopa_deal_rate     - efficiency: deals among games where a deal was possible
  nozopa_walk_rate   - discipline: no deal closed when v <= c (rational walk)
  capture            - mean share of the realized ZOPA surplus kept
  sucker_rate        - fraction of deals with NEGATIVE own payoff (overpaid /
                       sold below cost): the cardinal negotiation sin
  bluff_rate         - fraction of cheap-talk reservation claims that were
                       false by >= 10 (structured claims; regex for free text)
  bluff_effect       - mean opponent concession after this agent's bluff claims
                       vs after its honest claims (noisy, pooled)
  rounds_to_deal     - speed (early deals keep more of the discounted pie)
  concession_*       - mean offered price by own-proposal index, per role

Blotto:
  avg_share          - mean payoff share of the constant-sum 100
  rank_profile       - mean troops allocated by field-value rank (1=richest)
  rank_win_rate      - fraction of rank-r fields captured (winner or tie)
  underspend         - mean unspent troops (pure waste in this game)

Plus (both games): illegal_rate = rejected moves / decisions attempted.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

from ..textclaims import parse_claim as claimed_reservation_from_text  # noqa: F401 (re-exported)


def _round(x, n=3):
    return None if x is None else round(x, n)


# ---------------------------------------------------------------- bargaining
def bargaining_agent_metrics(results: Sequence[dict], agent: str) -> dict:
    games = [r for r in results if agent in r["seats"].values()]
    n = len(games)
    m = dict(agent=agent, games=n)
    if n == 0:
        return m
    seat_of = {id(r): next(s for s, a in r["seats"].items() if a == agent) for r in games}

    dealt = zopa_games = zopa_deals = nozopa_games = nozopa_nodeals = 0
    sucker = 0
    captures: List[float] = []
    payoffs: List[float] = []
    rounds_to_deal: List[int] = []
    claims_made = bluffs = 0
    conc_after_bluff: List[float] = []
    conc_after_honest: List[float] = []
    concession_buyer: Dict[int, List[float]] = {}
    concession_seller: Dict[int, List[float]] = {}
    dec_count = illegal_count = 0

    for r in games:
        seat = seat_of[id(r)]
        opp_seat = "seller" if seat == "buyer" else "buyer"
        t = r["truth"]
        v, c, zopa, has_zopa = t["v"], t["c"], t["zopa"], t["has_zopa"]
        pay = r["payoffs"][seat]
        payoffs.append(pay)
        is_deal = t["deal"] is not None
        dealt += int(is_deal)
        zopa_games += int(has_zopa)
        zopa_deals += int(has_zopa and is_deal)
        nozopa_games += int(not has_zopa)
        nozopa_nodeals += int(not has_zopa and not is_deal)
        if is_deal:
            rounds_to_deal.append(t["deal"]["t"])
            if pay < -1e-9:
                sucker += 1
            if zopa > 0:
                captures.append(pay / zopa)
        zopa_games_flag = has_zopa
        # per-decision scans: claims, bluffs, concession curve, illegal moves
        own_offers: List[dict] = []
        own_idx = 0
        events_offers = [e for e in r["events"] if e["type"] == "offer"]
        opp_offer_seq = [e for e in events_offers if e["by"] == opp_seat]
        for d in r["decisions"]:
            if d["agent"] != agent:
                continue
            dec_count += 1
            if d.get("belief", {}).get("illegal_move"):
                illegal_count += 1
            claims = d.get("belief", {}).get("claims")
            kind = claims.get("kind") if claims else None
            claimed = claims.get("reservation") if claims else None
            if claimed is None and d.get("message"):
                claimed = claimed_reservation_from_text(d["message"])
                kind = "text"
            if d["kind"] == "propose" and claimed is not None:
                claims_made += 1
                true_res = v if seat == "buyer" else c
                is_bluff = (claimed <= true_res - 10) if seat == "buyer" else (claimed >= true_res + 10)
                if is_bluff:
                    bluffs += 1
                # opponent's next-offer concession after this claim
                nxt = _next_opp_offer_move(events_offers, d, seat, opp_seat)
                if nxt is not None:
                    (conc_after_bluff if is_bluff else conc_after_honest).append(nxt)
            if d["kind"] == "propose" and zopa_games_flag:
                own_idx += 1
                bucket = concession_buyer if seat == "buyer" else concession_seller
                bucket.setdefault(own_idx, []).append(d["params"]["price"])

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    m.update(
        avg_payoff=_round(_mean(payoffs), 2),
        deal_rate=_round(dealt / n),
        zopa_deal_rate=_round(zopa_deals / zopa_games) if zopa_games else None,
        nozopa_walk_rate=_round(nozopa_nodeals / nozopa_games) if nozopa_games else None,
        capture=_round(_mean(captures)) if captures else None,
        sucker_rate=_round(sucker / dealt) if dealt else None,
        rounds_to_deal=_round(_mean(rounds_to_deal), 2) if rounds_to_deal else None,
        bluff_rate=_round(bluffs / claims_made) if claims_made else None,
        claims=claims_made, bluffs=bluffs,
        conc_after_bluff=_round(_mean(conc_after_bluff), 2) if conc_after_bluff else None,
        conc_after_honest=_round(_mean(conc_after_honest), 2) if conc_after_honest else None,
        illegal_rate=_round(illegal_count / dec_count) if dec_count else 0.0,
        concession_buyer={k: _round(_mean(vv), 1) for k, vv in sorted(concession_buyer.items())},
        concession_seller={k: _round(_mean(vv), 1) for k, vv in sorted(concession_seller.items())},
        zopa_games=zopa_games, nozopa_games=nozopa_games,
    )
    return m


def _proposal_index(r: dict, decision: dict) -> int:
    """1-based index of this decision among the agent's own proposals."""
    k = 0
    for d in r["decisions"]:
        if d["agent"] == decision["agent"] and d["kind"] == "propose":
            k += 1
            if d is decision or (d["step"] == decision["step"] and d["seat"] == decision["seat"]):
                return k
    return k


def _next_opp_offer_move(events_offers: List[dict], decision: dict,
                         seat: str, opp_seat: str):
    """How far the opponent's next offer moved toward `seat` after `decision`.
    Positive = concession favorable to `seat`."""
    step = decision["step"]
    # find opponent offers that came after this decision, and the one before
    before = [e for e in events_offers if e["by"] == opp_seat and e["t"] <= step]
    after = [e for e in events_offers if e["by"] == opp_seat and e["t"] > step]
    if not after or not before:
        return None
    p_prev, p_new = before[-1]["price"], after[0]["price"]
    return (p_prev - p_new) if seat == "buyer" else (p_new - p_prev)


# ------------------------------------------------------------------- blotto
def blotto_agent_metrics(results: Sequence[dict], agent: str) -> dict:
    games = [r for r in results if agent in r["seats"].values()]
    n = len(games)
    m = dict(agent=agent, games=n)
    if n == 0:
        return m
    shares: List[float] = []
    underspend: List[float] = []
    rank_troops: Dict[int, List[float]] = {}
    rank_wins: Dict[int, List[float]] = {}
    dec_count = illegal_count = 0
    for r in games:
        seat = next(s for s, a in r["seats"].items() if a == agent)
        t = r["truth"]
        if not t.get("allocations"):
            continue
        shares.append(r["payoffs"][seat] / 100.0)
        alloc = t["allocations"][seat]
        underspend.append(60 - sum(alloc))
        values = t["values"]
        order = sorted(range(len(values)), key=lambda i: -values[i])
        for rank, field_idx in enumerate(order, start=1):
            rank_troops.setdefault(rank, []).append(alloc[field_idx])
            res = t["field_results"][field_idx]
            won = res["winner"] == seat or res["winner"] == "tie"
            rank_wins.setdefault(rank, []).append(1.0 if won else 0.0)
        for d in r["decisions"]:
            if d["agent"] == agent:
                dec_count += 1
                if d.get("belief", {}).get("illegal_move"):
                    illegal_count += 1

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    m.update(
        avg_share=_round(_mean(shares)),
        avg_payoff=_round(100 * _mean(shares), 2) if shares else None,
        underspend=_round(_mean(underspend), 2) if underspend else None,
        rank_profile={k: _round(_mean(vv), 2) for k, vv in sorted(rank_troops.items())},
        rank_win_rate={k: _round(_mean(vv)) for k, vv in sorted(rank_wins.items())},
        illegal_rate=_round(illegal_count / dec_count) if dec_count else 0.0,
    )
    return m


def all_metrics(game_name: str, results: Sequence[dict], agents: Sequence[str]) -> List[dict]:
    fn = bargaining_agent_metrics if game_name == "bargaining" else blotto_agent_metrics
    return [fn(results, a) for a in agents]
