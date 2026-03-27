"""Core trading engine — Section 10.

Runs on every odds update from the Betfair stream. Evaluates signals,
places entries, monitors positions, and triggers exits.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import Config
from .market_filter import MarketFilter
from .models import (
    ExitReason,
    MarketState,
    MarketStatus,
    Position,
    ScoreSource,
    ScoreState,
    Side,
)
from .positions import PositionTracker
from .probability import calculate_model_odds, calculate_player1_win_prob
from .risk import RiskManager
from .ticks import nearest_tick

log = logging.getLogger(__name__)


class TradingEngine:
    """Core trading logic — evaluates signals and manages trade lifecycle."""

    def __init__(
        self,
        config: Config,
        positions: PositionTracker,
        risk: RiskManager,
        market_filter: MarketFilter,
    ) -> None:
        self.config = config
        self.positions = positions
        self.risk = risk
        self.market_filter = market_filter

        # Callback for logging trades
        self.on_trade_closed: Optional[callable] = None
        # Callback for alerts
        self.on_alert: Optional[callable] = None

    async def on_market_update(
        self,
        market_id: str,
        market: MarketState,
        scores: dict[str, ScoreState],
    ) -> None:
        """Called on every odds update from the Betfair stream."""
        # Handle market suspension
        if market.status == MarketStatus.SUSPENDED:
            await self._handle_suspension(market_id)
            return

        # Find matching score state
        score = self._find_score(market_id, scores)

        # Check exit conditions for open position
        if market_id in self.positions.positions:
            await self._check_exit(market_id, market, score)
            return  # Don't look for new entries while in a position

        # Look for new entry signal
        if score is not None:
            await self._evaluate_entry(market_id, market, score)

    # ------------------------------------------------------------------
    # Entry evaluation (Section 10, Steps 1-8)
    # ------------------------------------------------------------------

    async def _evaluate_entry(
        self,
        market_id: str,
        market: MarketState,
        score: ScoreState,
    ) -> None:
        # Step 1: Model state freshness
        if not score.is_fresh:
            return

        # Step 2: Skip first 2 points of new game
        if score.points_in_current_game < self.config.new_game_skip_points:
            return

        # Step 6: Check for existing position
        if self.positions.has_position(market_id):
            return

        # Step 3: Calculate model odds
        p1_prob, p1_model_odds = calculate_player1_win_prob(score)

        # Get player1's selection ID and market odds
        sel_id = score.player1_selection_id
        if sel_id == 0:
            return

        runner = market.runners.get(sel_id)
        if runner is None:
            return

        market_odds = runner.best_back_price
        if market_odds <= 0:
            return

        # Market filter (Section 15)
        passes, reason = self.market_filter.qualifies(market, score, sel_id)
        if not passes:
            return

        # Step 4: Calculate edge
        edge = abs(market_odds - p1_model_odds) / p1_model_odds
        if edge < self.config.min_edge:
            return

        # Step 5: Check odds range
        if market_odds < self.config.min_odds or market_odds > self.config.max_odds:
            return

        # Determine side
        if market_odds > p1_model_odds:
            # Market overpricing player1 → BACK player1
            side = Side.BACK
            entry_price = runner.best_back_price
        else:
            # Market underpricing player1 → LAY player1
            side = Side.LAY
            entry_price = runner.best_lay_price

        if entry_price <= 0:
            return

        # Step 7: Risk manager checks
        is_inference = score.source == ScoreSource.INFERENCE
        stake = self.risk.calculate_stake(is_inference)

        # Check minimum net profit
        gross = stake * abs(entry_price - p1_model_odds) / p1_model_odds
        commission = gross * self.config.commission_rate
        net = gross - commission
        if net < self.config.min_net_profit:
            return

        approved, reject_reason = self.risk.check_trade(
            stake=stake,
            market_exposure=self.positions.market_exposure(market_id),
            portfolio_exposure=self.positions.total_exposure,
            is_inference=is_inference,
        )
        if not approved:
            log.debug("Trade rejected by risk: %s", reject_reason)
            return

        # Step 8: Place limit order
        score_str = (
            f"P:{score.point_score} G:{score.game_score} "
            f"S:{score.set_score}"
        )

        pos = await self.positions.place_entry(
            market_id=market_id,
            selection_id=sel_id,
            side=side,
            price=entry_price,
            stake=stake,
            model_odds=p1_model_odds,
            market_odds=market_odds,
            edge=edge,
            score_at_entry=score_str,
            model_state_age=score.age_seconds,
            score_source=score.source,
            event_name=market.event_name or f"{score.player1_name} v {score.player2_name}",
            tournament=score.tournament,
            surface=score.surface,
        )

        if pos:
            self.risk.recent_trades.append(pos.entry_time)
            log.info(
                "ENTRY: %s %s @ %.2f (model=%.2f, edge=%.1f%%) on %s",
                side.value, score.player1_name, entry_price,
                p1_model_odds, edge * 100, score.tournament,
            )

    # ------------------------------------------------------------------
    # Exit checks (Section 11)
    # ------------------------------------------------------------------

    async def _check_exit(
        self,
        market_id: str,
        market: MarketState,
        score: Optional[ScoreState],
    ) -> None:
        # Get model odds for edge-gone check
        model_odds = 0.0
        if score and score.is_fresh:
            _, model_odds = calculate_player1_win_prob(score)

        result = self.positions.check_exit(market_id, market, model_odds)
        if result is None:
            return

        reason, current_price = result
        use_limit = reason == ExitReason.TARGET_REACHED

        # Calculate target price for limit exit
        pos = self.positions.positions.get(market_id)
        target_price = 0.0
        if use_limit and pos:
            if pos.side == Side.BACK:
                target_price = current_price  # lay at current
            else:
                target_price = current_price

        closed = await self.positions.close_position(
            market_id, current_price, reason,
            use_limit=use_limit, target_price=target_price,
        )

        if closed and pos:
            exit_odds, net_profit = closed
            won = net_profit > 0
            self.risk.record_trade(net_profit, won)

            log.info(
                "EXIT (%s): %s @ %.2f → %.2f, net £%.2f, held %.1fs",
                reason.value, pos.side.value, pos.entry_odds,
                exit_odds, net_profit, pos.hold_seconds,
            )

            # Trigger trade log callback
            if self.on_trade_closed:
                await self.on_trade_closed(pos, exit_odds, net_profit, reason)

            # Alert on large loss
            if net_profit < -30 and self.on_alert:
                await self.on_alert(
                    "warning",
                    f"Large loss: £{net_profit:.2f} on {pos.event_name} "
                    f"({reason.value})",
                )

    # ------------------------------------------------------------------
    # Suspension handling (Section 14)
    # ------------------------------------------------------------------

    async def _handle_suspension(self, market_id: str) -> None:
        """Handle SUSPENDED market status."""
        await self.positions.cancel_all_orders(market_id)
        # Hold position — will close when market returns to OPEN

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_score(
        self,
        market_id: str,
        scores: dict[str, ScoreState],
    ) -> Optional[ScoreState]:
        """Find the ScoreState mapped to this Betfair market."""
        for state in scores.values():
            if state.betfair_market_id == market_id:
                return state
        return None
