"""Betfair tick-size ladder and price utilities."""

from __future__ import annotations

import bisect
from functools import lru_cache

# Build the full tick ladder once at import time.
_TICK_BANDS: list[tuple[float, float, float]] = [
    (1.01, 2.00, 0.01),
    (2.00, 3.00, 0.02),
    (3.00, 4.00, 0.05),
    (4.00, 6.00, 0.10),
    (6.00, 10.00, 0.20),
    (10.00, 20.00, 0.50),
    (20.00, 30.00, 1.00),
]


def _build_ladder() -> list[float]:
    ladder: list[float] = []
    for lo, hi, step in _TICK_BANDS:
        price = lo
        while price < hi - step / 2:
            ladder.append(round(price, 2))
            price += step
    ladder.append(round(_TICK_BANDS[-1][1], 2))  # include 30.00
    return ladder


LADDER: list[float] = _build_ladder()


def nearest_tick(price: float) -> float:
    """Round *price* to the nearest valid Betfair tick."""
    idx = bisect.bisect_left(LADDER, price)
    if idx == 0:
        return LADDER[0]
    if idx >= len(LADDER):
        return LADDER[-1]
    lo, hi = LADDER[idx - 1], LADDER[idx]
    return lo if (price - lo) <= (hi - price) else hi


def tick_index(price: float) -> int:
    """Return the index of *price* in the tick ladder."""
    idx = bisect.bisect_left(LADDER, price - 1e-9)
    if idx < len(LADDER) and abs(LADDER[idx] - price) < 1e-9:
        return idx
    return bisect.bisect_left(LADDER, price)


def ticks_between(price_a: float, price_b: float) -> int:
    """Signed tick distance from *price_a* to *price_b*."""
    return tick_index(price_b) - tick_index(price_a)


def move_ticks(price: float, n: int) -> float:
    """Move *n* ticks from *price* (positive = higher odds)."""
    idx = tick_index(price) + n
    idx = max(0, min(idx, len(LADDER) - 1))
    return LADDER[idx]


@lru_cache(maxsize=512)
def tick_increment_at(price: float) -> float:
    """Return the tick increment for a given price level."""
    for lo, hi, step in _TICK_BANDS:
        if lo <= price < hi:
            return step
    return _TICK_BANDS[-1][2]


def spread_in_ticks(back: float, lay: float) -> int:
    """Number of ticks between best back and best lay."""
    return abs(ticks_between(back, lay))
