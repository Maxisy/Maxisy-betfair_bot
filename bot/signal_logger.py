"""Signal logger for dry-run edge validation.

In dry-run mode, logs two streams of data:

1. signals.jsonl — every time the model detects a potential edge, logged with
   timestamp, model odds, market odds, score state, and what the bot WOULD do.

2. odds_tape.jsonl — continuous log of market odds + model odds for every
   active match on every odds tick. This is the raw tape used by the analysis
   script to check what happened AFTER each signal.

Together these let you answer: "If I had placed this trade at this moment,
what would the market odds have been 5/10/30/60 seconds later?"
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import MarketState, ScoreSource, ScoreState
from .probability import calculate_player1_win_prob
from .ticks import nearest_tick, spread_in_ticks

log = logging.getLogger(__name__)

SIGNALS_FILE = Path("data/signals.jsonl")
ODDS_TAPE_FILE = Path("data/odds_tape.jsonl")

# Throttle odds tape: max one write per market per N seconds
TAPE_INTERVAL = 1.0


class SignalLogger:
    """Logs signals and continuous odds tape for post-hoc edge analysis."""

    def __init__(self) -> None:
        SIGNALS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._last_tape_write: dict[str, float] = {}  # market_id -> timestamp
        self._signal_count = 0
        self._tape_count = 0

    def log_signal(
        self,
        market_id: str,
        market: MarketState,
        score: ScoreState,
        selection_id: int,
        side: str,
        model_odds: float,
        market_odds: float,
        edge: float,
        best_back: float,
        best_lay: float,
    ) -> None:
        """Log a signal — a moment the bot WOULD have entered a trade."""
        now = time.time()
        self._signal_count += 1

        entry = {
            "type": "signal",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_ts": round(now, 3),
            "signal_id": self._signal_count,
            "market_id": market_id,
            "selection_id": selection_id,
            "event_name": market.event_name,
            "tournament": score.tournament,
            "surface": score.surface,
            "player1": score.player1_name,
            "player2": score.player2_name,
            "side": side,
            "model_odds": round(model_odds, 4),
            "market_back": round(best_back, 2),
            "market_lay": round(best_lay, 2),
            "market_mid": round((best_back + best_lay) / 2, 4) if best_lay > 0 else 0,
            "spread_ticks": spread_in_ticks(best_back, best_lay) if best_lay > best_back else 0,
            "edge_pct": round(edge * 100, 2),
            "point_score": list(score.point_score),
            "game_score": list(score.game_score),
            "set_score": list(score.set_score),
            "server": score.server,
            "p1_serve_pct": round(score.player1_serve_pct, 4),
            "p2_serve_pct": round(score.player2_serve_pct, 4),
            "score_age_sec": round(score.age_seconds, 1),
            "score_source": score.source.value,
            "total_matched": round(market.total_matched, 0),
        }

        self._append(SIGNALS_FILE, entry)
        log.info(
            "SIGNAL #%d: %s %s @ back=%.2f lay=%.2f (model=%.2f, edge=%.1f%%) %s",
            self._signal_count, side, score.player1_name,
            best_back, best_lay, model_odds, edge * 100, score.tournament,
        )

    def log_odds_tick(
        self,
        market_id: str,
        market: MarketState,
        score: Optional[ScoreState],
    ) -> None:
        """Log an odds tick to the continuous tape.

        Throttled to one write per market per second to avoid huge files.
        """
        now = time.time()
        last = self._last_tape_write.get(market_id, 0)
        if now - last < TAPE_INTERVAL:
            return
        self._last_tape_write[market_id] = now

        # Need at least one runner with prices
        if not market.runners:
            return

        # Calculate model odds if score available
        model_odds_p1 = 0.0
        model_prob_p1 = 0.0
        if score and score.is_fresh and score.player1_selection_id:
            try:
                model_prob_p1, model_odds_p1 = calculate_player1_win_prob(score)
            except Exception:
                pass

        # Build runner snapshot
        runners: dict[str, Any] = {}
        for sel_id, runner in market.runners.items():
            runners[str(sel_id)] = {
                "back": round(runner.best_back_price, 2),
                "back_size": round(runner.best_back_size, 2),
                "lay": round(runner.best_lay_price, 2),
                "lay_size": round(runner.best_lay_size, 2),
                "ltp": round(runner.last_traded_price, 2),
            }

        entry = {
            "type": "tick",
            "unix_ts": round(now, 3),
            "market_id": market_id,
            "event_name": market.event_name,
            "in_play": market.in_play,
            "status": market.status.value,
            "runners": runners,
            "model_odds_p1": round(model_odds_p1, 4) if model_odds_p1 else None,
            "model_prob_p1": round(model_prob_p1, 4) if model_prob_p1 else None,
        }

        # Add score if available
        if score:
            entry.update({
                "point_score": list(score.point_score),
                "game_score": list(score.game_score),
                "set_score": list(score.set_score),
                "server": score.server,
                "score_age_sec": round(score.age_seconds, 1),
            })

        self._tape_count += 1
        self._append(ODDS_TAPE_FILE, entry)

    def _append(self, path: Path, data: dict) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
        except Exception as e:
            log.error("Signal logger write failed (%s): %s", path, e)

    @property
    def stats(self) -> dict[str, int]:
        return {"signals": self._signal_count, "tape_ticks": self._tape_count}
