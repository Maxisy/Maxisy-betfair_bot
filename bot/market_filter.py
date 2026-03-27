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

        # Tournament filter — exclude Grand Slams and Masters 1000
        if self._is_excluded_tournament(score.tournament):
            return self._reject(market.market_id, "excluded_tournament")

        return True, ""

    def _is_excluded_tournament(self, tournament: str) -> bool:
        tournament_lower = tournament.lower()
        for excluded in self.config.excluded_tournaments:
            if excluded.lower() in tournament_lower:
                return True
        return False

    def _reject(self, market_id: str, reason: str) -> tuple[bool, str]:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return False, reason
