"""Game runner + round-robin tournament with duplicate deals.

Duplicate-deal design (borrowed from bridge): every agent pair plays the
*same* list of scenario seeds, with seats swapped. This removes scenario
luck from head-to-head comparisons - if pair (A,B) and pair (C,D) both play
seed 17, they face identical private values/layouts.

Engine-exploitation guard: every agent move goes through game.validate();
illegal moves are replaced by game.fallback_action() and counted per agent.
A tournament where some agent has a high illegal rate is a red flag that the
agent (or its LLM) is trying to operate outside the rules.
"""
from __future__ import annotations

import itertools
import time
from typing import Callable, Dict, List

from ..agents.base import Agent, Decision
from ..games.base import Game


def play_game(game: Game, agents_by_seat: Dict[str, Agent]) -> dict:
    seats = game.seats()
    assert set(agents_by_seat) == set(seats), f"need agents for {seats}"
    reset_public = {"seed": game.seed, "_game_ref": game, **game.public_view()}
    for seat in seats:
        agents_by_seat[seat].reset(seat, game.private_view(seat), dict(reset_public))

    decisions: List[dict] = []
    illegal: Dict[str, int] = {}
    events: List[dict] = []
    t0 = time.time()
    guard = 0
    while not game.is_terminal() and guard < 128:
        guard += 1
        acting = game.acting_seats()
        if not acting:
            break
        actions: Dict[str, object] = {}
        entries: List[dict] = []
        for seat in acting:
            agent = agents_by_seat[seat]
            hint = game.action_hint(seat)
            dec: Decision = agent.act(game.public_view(), game.private_view(seat), hint)
            norm, err = game.validate(seat, dec.action)
            if norm is None:
                illegal[agent.name] = illegal.get(agent.name, 0) + 1
                fb = game.fallback_action(seat)
                norm, err2 = game.validate(seat, fb)
                assert norm is not None, f"engine fallback illegal ({err2}) - engine bug"
                dec = Decision(norm, f"[ILLEGAL MOVE REJECTED: {err}] {dec.reasoning}",
                               {**dec.belief, "illegal_move": err})
            actions[seat] = norm
            entry = {"seat": seat, "agent": agent.name, "step": guard}
            entry.update(dec.to_dict())
            entry.pop("_game_ref", None)
            entries.append(entry)
        events.extend(game.step(actions))
        decisions.extend(entries)

    return {
        "game": game.name,
        "seed": game.seed,
        "seats": {seat: agents_by_seat[seat].name for seat in seats},
        "decisions": decisions,
        "events": events,
        "payoffs": game.payoffs(),
        "truth": game.truth(),
        "illegal": illegal,
        "steps": guard,
        "wall_s": round(time.time() - t0, 4),
    }


def run_tournament(game_cls: Callable[[int], Game],
                   agent_factories: Dict[str, Callable[[], Agent]],
                   seeds: List[int], swap_seats: bool = True,
                   progress_every: int = 0) -> List[dict]:
    """Full round-robin: every unordered pair x every seed x both seat orders."""
    results: List[dict] = []
    names = list(agent_factories)
    seat_order = game_cls(seeds[0]).seats()
    flips = (False, True) if swap_seats and len(seat_order) == 2 else (False,)
    total = len(names) * (len(names) - 1) // 2 * len(seeds) * len(flips)
    done = 0
    for a, b in itertools.combinations(names, 2):
        for seed in seeds:
            for flip in flips:
                if not flip:
                    assign = {seat_order[0]: a, seat_order[1]: b}
                else:
                    assign = {seat_order[0]: b, seat_order[1]: a}
                agents = {seat: agent_factories[n]() for seat, n in assign.items()}
                res = play_game(game_cls(seed), agents)
                res["pair"] = [a, b]
                results.append(res)
                done += 1
                if progress_every and done % progress_every == 0:
                    print(f"  [{game_cls(seeds[0]).name}] {done}/{total} games", flush=True)
    return results
