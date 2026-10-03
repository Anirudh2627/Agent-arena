"""Shared cheap-talk claim extraction.

Used by (a) the analysis layer to measure bluff rates from free-form LLM
messages and (b) the Gullible baseline agent - the one scripted opponent that
actually *reads* messages, which is what puts real selection pressure on the
learned bluff knob during self-play.
"""
from __future__ import annotations

import re
from typing import Optional

CLAIM_RE = re.compile(
    r"(?:my|our)\s+(?:absolute\s+|real\s+|true\s+|honest\s+)?"
    r"(?:ceiling|floor|limit|budget|value|cost|reservation|maximum|minimum|walkaway|walk-away)"
    r"\s+(?:is|are|:|of|caps at|sits at|maxes out at|stands at)?\s*"
    r"(?:about\s+|around\s+|only\s+|just\s+|roughly\s+)?(\d+)",
    re.IGNORECASE)


def parse_claim(text: str) -> Optional[int]:
    """Extract the first stated reservation number from a message, if any."""
    if not text:
        return None
    m = CLAIM_RE.search(text)
    return int(m.group(1)) if m else None
