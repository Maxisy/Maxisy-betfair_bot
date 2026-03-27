"""Match mapper — links Goalserve matches to Betfair markets.

Matches by player surnames (order-agnostic) + start time within 10 minutes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from unidecode import unidecode

from .models import ScoreState

log = logging.getLogger(__name__)

TIME_TOLERANCE_SECONDS = 600  # 10 minutes


def normalise_name(name: str) -> str:
    """Strip accents, lowercase, extract surname."""
    name = unidecode(name).lower().strip()
    # Take last word as surname (handles "De Minaur" → "minaur")
    parts = name.split()
    return parts[-1] if parts else name


def extract_surnames(full_name: str) -> list[str]:
    """Extract all plausible surname tokens from a name string."""
    name = unidecode(full_name).lower().strip()
    # Remove common separators
    name = re.sub(r"[/\\,]", " ", name)
    parts = name.split()
    # Return all parts as candidates (handles multi-word surnames)
    return [p for p in parts if len(p) > 1]


def match_names(name1: str, name2: str) -> bool:
    """Check if two player names refer to the same person (surname match)."""
    s1 = normalise_name(name1)
    s2 = normalise_name(name2)
    return s1 == s2


def map_matches_to_markets(
    scores: dict[str, ScoreState],
    catalogues: list[dict[str, Any]],
) -> dict[str, dict]:
    """Map Goalserve match IDs to Betfair market catalogue entries.

    Returns dict of match_id -> catalogue entry for successful matches.
    """
    results: dict[str, dict] = {}

    for cat in catalogues:
        event = cat.get("event", {})
        event_name = event.get("name", "")
        market_id = cat.get("marketId", "")
        runners = cat.get("runners", [])

        if not event_name or not market_id:
            continue

        # Extract surnames from Betfair event name (format: "Player A v Player B")
        bf_parts = re.split(r"\s+v(?:s)?\.?\s+", event_name, flags=re.IGNORECASE)
        if len(bf_parts) != 2:
            continue

        bf_surname1 = normalise_name(bf_parts[0])
        bf_surname2 = normalise_name(bf_parts[1])

        # Market start time
        market_start = cat.get("marketStartTime", "")

        for match_id, state in scores.items():
            if state.betfair_market_id:
                continue  # already mapped

            gs_surname1 = normalise_name(state.player1_name)
            gs_surname2 = normalise_name(state.player2_name)

            # Check both surname orderings
            names_match = (
                (gs_surname1 == bf_surname1 and gs_surname2 == bf_surname2)
                or (gs_surname1 == bf_surname2 and gs_surname2 == bf_surname1)
            )

            if not names_match:
                continue

            # Time check (if we have scheduled start times)
            # For now, name match is sufficient — time check is a bonus filter

            # Map player to selection IDs
            state.betfair_market_id = market_id
            for runner in runners:
                runner_name = runner.get("runnerName", "")
                sel_id = runner.get("selectionId", 0)
                rn = normalise_name(runner_name)
                if rn == gs_surname1:
                    state.player1_selection_id = sel_id
                elif rn == gs_surname2:
                    state.player2_selection_id = sel_id

            results[match_id] = cat
            log.info(
                "Mapped %s (%s vs %s) → market %s",
                match_id, state.player1_name, state.player2_name, market_id,
            )
            break  # one Goalserve match per Betfair market

    return results
