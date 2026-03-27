"""Model tape logger for dry-run edge validation.

In paper mode, logs model_tape.jsonl: model odds + score state saved every
second for each active match. After the match, this is merged with real
Betfair historical data (not delayed) by backtest_match.py to find true edges.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .models import ScoreState
from .probability import calculate_player1_win_prob

log = logging.getLogger(__name__)

MODEL_TAPE_FILE = Path("data/model_tape.jsonl")

# Max one write per market per second
TAPE_INTERVAL = 1.0


class SignalLogger:
    """Logs model calculations for post-match backtesting against real odds."""

    def __init__(self) -> None:
        MODEL_TAPE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._last_model_write: dict[str, float] = {}
        self._model_count = 0

    def log_model_tick(
        self,
        market_id: str,
        score: ScoreState,
    ) -> None:
        """Log model odds + score state every second per market.

        This is independent of Betfair market data — purely what the model
        thinks at each moment. After the match, this tape is merged with
        real (non-delayed) Betfair historical data to find true edges.
        """
        now = time.time()
        last = self._last_model_write.get(market_id, 0)
        if now - last < TAPE_INTERVAL:
            return
        self._last_model_write[market_id] = now

        if not score.is_fresh or not score.player1_selection_id:
            return

        try:
            model_prob_p1, model_odds_p1 = calculate_player1_win_prob(score)
        except Exception:
            return

        entry = {
            "type": "model",
            "unix_ts": round(now, 3),
            "market_id": market_id,
            "selection_id": score.player1_selection_id,
            "player1": score.player1_name,
            "player2": score.player2_name,
            "tournament": score.tournament,
            "surface": score.surface,
            "model_odds_p1": round(model_odds_p1, 4),
            "model_prob_p1": round(model_prob_p1, 4),
            "point_score": list(score.point_score),
            "game_score": list(score.game_score),
            "set_score": list(score.set_score),
            "server": score.server,
            "p1_serve_pct": round(score.player1_serve_pct, 4),
            "p2_serve_pct": round(score.player2_serve_pct, 4),
            "score_age_sec": round(score.age_seconds, 1),
        }

        self._model_count += 1
        self._append(entry)

    def _append(self, data: dict) -> None:
        try:
            with open(MODEL_TAPE_FILE, "a") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
        except Exception as e:
            log.error("Model tape write failed: %s", e)

    @property
    def stats(self) -> dict[str, int]:
        return {"model_ticks": self._model_count}
