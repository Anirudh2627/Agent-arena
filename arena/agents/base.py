"""Agent abstractions: the Decision record and the Agent interface.

Every agent - scripted baseline or LLM-driven - speaks the same protocol:

    reset(seat, private, public)   once per game
    act(public, private, hint) -> Decision(action, reasoning, belief)

``belief`` is a free-form JSON dict persisted into the replay; scripted
agents put posterior grids / claimed reservations there, LLM agents put
their self-reported belief state. The replay viewer renders whatever is
in ``belief.posterior`` as a histogram.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..games.base import Action


@dataclass
class Decision:
    action: Action
    reasoning: str = ""
    belief: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"kind": self.action.kind,
                "params": dict(self.action.params),
                "message": self.action.message,
                "reasoning": self.reasoning,
                "belief": self.belief}


class Agent:
    name = "agent"
    family = "abstract"       # "scripted" | "llm" | "mock"

    def __init__(self, name: str | None = None):
        if name:
            self.name = name
        self.rng = None       # per-game rng, set in reset()
        self.seat = None

    def reset(self, seat: str, private: dict, public: dict) -> None:
        import random
        self.seat = seat
        self.private = private
        self.rng = random.Random(f"{self.name}:{seat}:{public.get('seed', 0)}")

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        raise NotImplementedError


class IllegalAgent(Agent):
    """Test agent that deliberately submits illegal moves. Used by the test
    suite to verify the engine's validation + fallback path (the
    'how did you prevent agents from exploiting the engine?' story)."""
    name = "illegal_probe"
    family = "adversarial"

    def act(self, public: dict, private: dict, hint: dict) -> Decision:
        if public.get("game") == "bargaining":
            return Decision(Action("propose", {"price": 99999}, "I own the engine."),
                            reasoning="attempt to submit an out-of-range price")
        return Decision(Action("allocate", {"troops": [999] * 5}),
                        reasoning="attempt to overspend the budget")
