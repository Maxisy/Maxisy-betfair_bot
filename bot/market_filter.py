"""Market selection filter — Section 15.

Evaluates every market against qualifying criteria before passing to trading logic.
"""

from __future__ import annotations

import logging

from .config import Config
from .models import MarketState, MarketStatus, ScoreState
from .ticks import spread_in_ticks

log = logging.getLogger(__name__)


class MarketFilter:
    def __init__(self, config: Config) -> None:
        self.config = config
        # Rejection counters for daily stats
        self.rejections: dict[str, int] = {}

    def reset_daily(self) -> None:
        self.rejections.clear()

    def qualifies(
        self,
        market: MarketState,
        score: ScoreState | None,
        selection_id: int,
    ) -> tuple[bool, str]:
        """Check if a market + selection qualifies for trading.

        Returns (passes, reason) where reason is empty string on pass.
        """
        # Market must be in-play
        if not market.in_play:
            return self._reject(market.market_id, "not_in_play")

        # Market must be OPEN
        if market.status != MarketStatus.OPEN:
            return self._reject(market.market_id, "not_open")

        # Total matched volume
        if market.total_matched < self.config.min_matched_volume:
            return self._reject(market.market_id, "low_volume")

        # Runner data required
        runner = market.runners.get(selection_id)
        if runner is None:
            return self._reject(market.market_id, "no_runner_data")

        # Best back liquidity
        if runner.best_back_size < self.config.min_back_liquidity:
            return self._reject(market.market_id, "low_back_liquidity")

        # Odds range
        back = runner.best_back_price
        lay = runner.best_lay_price
        if back < self.config.min_odds or back > self.config.max_odds:
            return self._reject(market.market_id, "odds_out_of_range")
        if lay < self.config.min_odds or lay > self.config.max_odds:
            return self._reject(market.market_id, "odds_out_of_range")

        # Spread check
        spread = spread_in_ticks(back, lay)
        if spread < self.config.min_spread_ticks:
            return self._reject(market.market_id, "spread_too_tight")
        if spread > self.config.max_spread_ticks:
            return self._reject(market.market_id, "spread_too_wide")

        # Model state freshness
        if score is None:
            return self._reject(market.market_id, "no_score_state")
        if not score.is_fresh:
            return self._reject(market.market_id, "stale_model_state")

        # Tournament filter — must match allowed list and not be excluded
        if not self._is_allowed_tournament(score.tournament):
            return self._reject(market.market_id, "tournament_not_allowed")

        # Require real TA serve data for both players — no trading on defaults
        if score.player1_serve_pct < 0.01 or score.player2_serve_pct < 0.01:
            return self._reject(market.market_id, "missing_serve_data")

        # Only trade matches tracked from the start — no mid-match joins
        # without full hold/break history
        if not score.tracked_from_start:
            return self._reject(market.market_id, "joined_mid_match")

        # Require minimum service games so in-match adjustments have data
        total_svc = score.player1_service_games + score.player2_service_games
        if total_svc < self.config.min_service_games:
            return self._reject(market.market_id, "too_few_service_games")

        return True, ""

    def _is_allowed_tournament(self, tournament: str) -> bool:
        tournament_lower = tournament.lower()
        # First check exclusions (overrides allowed)
        for excluded in self.config.excluded_tournaments:
            if excluded.lower() in tournament_lower:
                return False
        # Then check if it matches any allowed tier
        for allowed in self.config.allowed_tournaments:
            if allowed.lower() in tournament_lower:
                return True
        return False

    def _reject(self, market_id: str, reason: str) -> tuple[bool, str]:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return False, reason
