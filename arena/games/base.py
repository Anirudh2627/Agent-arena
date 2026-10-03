"""Core game-engine abstractions for AgentArena.

Design rules (these matter for evaluation integrity):

1. **Information hiding.** Agents only ever receive ``public_view()`` plus their
   own ``private_view(seat)``. The full ground truth (``truth()``) is reserved
   for the analysis layer and is never passed to an agent at runtime.
2. **Legal-move validation.** Every action passes through ``validate()``, which
   normalizes/clamps parameters into the legal space and rejects anything else.
   The runner substitutes ``fallback_action()`` for rejected moves and records an
   illegality counter per agent (an anti-exploitation metric).
3. **Determinism.** All randomness derives from the scenario seed, so any pair
   of agents can be replayed exactly, and every pair in a tournament can play
   *the same* set of scenarios (duplicate-deals design, like bridge).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Action:
    """A normalized, JSON-serializable move."""
    kind: str                                    # e.g. "propose", "accept", "walk", "allocate"
    params: Dict[str, Any] = field(default_factory=dict)
    message: str = ""                            # cheap talk (untrusted by the engine)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": dict(self.params), "message": self.message}


class Game(ABC):
    """Interface implemented by every game engine (bargaining, Blotto, ...)."""

    name: str = "abstract"
    simultaneous: bool = False   # True -> all seats act in one sealed step

    def __init__(self, seed: int):
        self.seed = int(seed)

    # ---- topology -------------------------------------------------------
    @abstractmethod
    def seats(self) -> List[str]: ...

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def acting_seats(self) -> List[str]:
        """Seats that must submit an action on the current step."""

    # ---- views (information hiding boundary) ----------------------------
    @abstractmethod
    def public_view(self) -> Dict[str, Any]: ...

    @abstractmethod
    def private_view(self, seat: str) -> Dict[str, Any]: ...

    @abstractmethod
    def truth(self) -> Dict[str, Any]:
        """Full ground truth. Analysis/export only - NEVER shown to agents."""

    # ---- moves -----------------------------------------------------------
    @abstractmethod
    def action_hint(self, seat: str) -> Dict[str, Any]:
        """Description of the legal action space (kinds, ranges) for `seat`."""

    @abstractmethod
    def validate(self, seat: str, action: Action) -> Tuple[Optional[Action], str]:
        """Return (normalized_action, "") if legal, else (None, reason)."""

    @abstractmethod
    def fallback_action(self, seat: str) -> Action:
        """A guaranteed-legal safe action used when an agent submits junk."""

    @abstractmethod
    def step(self, actions: Dict[str, Action]) -> List[Dict[str, Any]]:
        """Apply one step; return the list of public events generated."""

    # ---- outcome ---------------------------------------------------------
    @abstractmethod
    def payoffs(self) -> Dict[str, float]: ...

    # ---- misc ------------------------------------------------------------
    def scenario_id(self) -> Any:
        return self.seed

    def public_transcript(self) -> str:
        """Compact human-readable transcript (used in LLM prompts)."""
        return json.dumps(self.public_view(), separators=(",", ":"))
