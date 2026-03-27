"""Order manager and position tracker.

Manages the full order lifecycle: place, monitor, partial fill handling, exit.
Tracks all open positions and checks exit conditions every second.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .betfair_client import BetfairClient
from .config import Config
from .models import (
    ExitReason,
    MarketState,
    MarketStatus,
    Position,
    ScoreSource,
    Side,
)
from .ticks import move_ticks, nearest_tick, ticks_between

log = logging.getLogger(__name__)


class PositionTracker:
    """Tracks all open positions and manages order lifecycle."""

    def __init__(self, config: Config, client: BetfairClient) -> None:
        self.config = config
        self.client = client

        # market_id -> Position
        self.positions: dict[str, Position] = {}
        # market_id -> bet_id for pending entry orders
        self.pending_entries: dict[str, str] = {}

    def has_position(self, market_id: str) -> bool:
        return market_id in self.positions or market_id in self.pending_entries

    @property
    def total_exposure(self) -> float:
        return sum(p.stake for p in self.positions.values())

    def market_exposure(self, market_id: str) -> float:
        pos = self.positions.get(market_id)
        return pos.stake if pos else 0.0

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    async def place_entry(
        self,
        market_id: str,
        selection_id: int,
        side: Side,
        price: float,
        stake: float,
        model_odds: float,
        market_odds: float,
        edge: float,
        score_at_entry: str,
        model_state_age: float,
        score_source: ScoreSource,
        event_name: str,
        tournament: str,
        surface: str,
    ) -> Optional[Position]:
        """Place a limit entry order and track it."""
        price = nearest_tick(price)

        instruction = self.client.build_limit_order(
            selection_id=selection_id,
            side=side.value,
            price=price,
            size=stake,
        )

        result = await self.client.place_orders(market_id, [instruction])

        if result.get("status") != "SUCCESS":
            log.error("Entry order failed for %s: %s", market_id, result)
            return None

        reports = result.get("instructionReports", [])
        if not reports:
            return None

        report = reports[0]
        bet_id = report.get("betId", "")
        matched = report.get("sizeMatched", 0)

        pos = Position(
            market_id=market_id,
            selection_id=selection_id,
            side=side,
            entry_odds=price,
            stake=round(matched, 2) if matched > 0 else stake,
            model_odds_at_entry=model_odds,
            market_odds_at_entry=market_odds,
            edge_at_entry=edge,
            score_at_entry=score_at_entry,
            model_state_age_at_entry=model_state_age,
            score_source_at_entry=score_source,
            event_name=event_name,
            tournament=tournament,
            surface=surface,
        )

        if matched >= stake * 0.99:
            # Fully filled
            self.positions[market_id] = pos
            log.info("Entry filled: %s %s @ %.2f (£%.2f) on %s",
                     side.value, selection_id, price, pos.stake, market_id)
        elif matched > 0:
            # Partially filled — cancel remainder, keep filled portion
            await self._cancel_order(market_id, bet_id)
            pos.stake = round(matched, 2)
            self.positions[market_id] = pos
            log.info("Entry partially filled: £%.2f of £%.2f on %s",
                     matched, stake, market_id)
        else:
            # Not filled — track as pending, will timeout
            self.pending_entries[market_id] = bet_id
            self.positions[market_id] = pos
            # Schedule cancellation after timeout
            asyncio.create_task(self._entry_timeout(market_id, bet_id))

        return pos

    async def _entry_timeout(self, market_id: str, bet_id: str) -> None:
        """Cancel unfilled entry after timeout."""
        await asyncio.sleep(self.config.entry_fill_timeout)
        if market_id in self.pending_entries:
            await self._cancel_order(market_id, bet_id)
            del self.pending_entries[market_id]
            # Remove position if it was never filled
            pos = self.positions.get(market_id)
            if pos and pos.stake <= 0:
                del self.positions[market_id]
                log.info("Entry expired (unfilled) on %s", market_id)

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    async def close_position(
        self,
        market_id: str,
        current_price: float,
        reason: ExitReason,
        use_limit: bool = False,
        target_price: float = 0.0,
    ) -> Optional[tuple[float, float]]:
        """Close an open position. Returns (exit_odds, net_profit) or None."""
        pos = self.positions.get(market_id)
        if pos is None:
            return None

        # Cancel any resting exit order
        if pos.exit_order_id:
            await self._cancel_order(market_id, pos.exit_order_id)
            pos.exit_order_id = None

        # Determine closing side (opposite of entry)
        close_side = "LAY" if pos.side == Side.BACK else "BACK"

        if use_limit and target_price > 0:
            exit_price = nearest_tick(target_price)
            instruction = self.client.build_limit_order(
                selection_id=pos.selection_id,
                side=close_side,
                price=exit_price,
                size=pos.stake,
            )
        else:
            # Market order to close immediately
            instruction = self.client.build_market_order(
                selection_id=pos.selection_id,
                side=close_side,
                size=pos.stake,
            )
            exit_price = current_price

        result = await self.client.place_orders(market_id, [instruction])

        if result.get("status") != "SUCCESS":
            log.error("Exit order failed for %s: %s", market_id, result)
            # Force market close
            instruction = self.client.build_market_order(
                selection_id=pos.selection_id,
                side=close_side,
                size=pos.stake,
            )
            result = await self.client.place_orders(market_id, [instruction])

        reports = result.get("instructionReports", [])
        if reports:
            report = reports[0]
            matched = report.get("sizeMatched", 0)
            avg_price = report.get("averagePriceMatched", exit_price)

            if matched < pos.stake * 0.99 and use_limit:
                # Partial fill on limit exit — schedule market close for remainder
                bet_id = report.get("betId", "")
                pos.exit_order_id = bet_id
                pos.exit_limit_placed_at = time.time()
                asyncio.create_task(
                    self._exit_timeout(market_id, bet_id, pos.stake - matched, close_side)
                )
                return None  # Not fully closed yet

            exit_price = avg_price

        # Calculate P&L
        gross = self._calc_gross_profit(pos, exit_price)
        commission = max(0, gross) * self.config.commission_rate
        net = gross - commission

        # Remove position
        del self.positions[market_id]
        self.pending_entries.pop(market_id, None)

        log.info(
            "Closed %s on %s @ %.2f (%s) — net £%.2f",
            pos.side.value, market_id, exit_price, reason.value, net,
        )

        return exit_price, net

    async def _exit_timeout(
        self, market_id: str, bet_id: str, remaining: float, close_side: str,
    ) -> None:
        """If exit limit not filled within timeout, close remainder at market."""
        await asyncio.sleep(self.config.exit_fill_timeout)
        pos = self.positions.get(market_id)
        if pos is None:
            return
        if pos.exit_order_id == bet_id:
            await self._cancel_order(market_id, bet_id)
            instruction = self.client.build_market_order(
                selection_id=pos.selection_id,
                side=close_side,
                size=remaining,
            )
            await self.client.place_orders(market_id, [instruction])
            log.info("Exit timeout — closed remainder at market on %s", market_id)

    # ------------------------------------------------------------------
    # Exit condition checks
    # ------------------------------------------------------------------

    def check_exit(
        self,
        market_id: str,
        market: MarketState,
        model_odds: float,
    ) -> Optional[tuple[ExitReason, float]]:
        """Check exit conditions for an open position.

        Returns (reason, current_price) if should exit, else None.
        """
        pos = self.positions.get(market_id)
        if pos is None:
            return None

        runner = market.runners.get(pos.selection_id)
        if runner is None:
            return None

        # Current price depends on side
        if pos.side == Side.BACK:
            current_price = runner.best_lay_price  # exit by laying
        else:
            current_price = runner.best_back_price  # exit by backing

        if current_price <= 0:
            return None

        # Priority 1: Stop loss — 4 ticks against
        ticks_against = self._ticks_against(pos, current_price)
        if ticks_against >= self.config.stop_loss_ticks:
            return ExitReason.STOP_LOSS, current_price

        # Priority 2: Target profit — 3 ticks of reversion
        ticks_for = self._ticks_for(pos, current_price)
        if ticks_for >= self.config.target_profit_ticks:
            return ExitReason.TARGET_REACHED, current_price

        # Priority 3: Max hold time
        if pos.hold_seconds >= self.config.max_hold_seconds:
            return ExitReason.TIME_EXIT, current_price

        # Priority 4: Edge gone
        market_odds = runner.best_back_price
        if model_odds > 0 and market_odds > 0:
            edge = abs(market_odds - model_odds) / model_odds
            if edge < self.config.edge_gone_threshold:
                return ExitReason.EDGE_GONE, current_price

        return None

    def _ticks_against(self, pos: Position, current_price: float) -> int:
        """How many ticks the position has moved against us."""
        if pos.side == Side.BACK:
            # Backed high, need price to stay high or go higher. Lower = against.
            return max(0, ticks_between(current_price, pos.entry_odds))
        else:
            # Laid low, need price to stay low or go lower. Higher = against.
            return max(0, ticks_between(pos.entry_odds, current_price))

    def _ticks_for(self, pos: Position, current_price: float) -> int:
        """How many ticks the position has moved in our favour."""
        if pos.side == Side.BACK:
            # Backed: higher current price = profit
            return max(0, ticks_between(pos.entry_odds, current_price))
        else:
            # Laid: lower current price = profit
            return max(0, ticks_between(current_price, pos.entry_odds))

    def _calc_gross_profit(self, pos: Position, exit_price: float) -> float:
        """Calculate gross profit for a closed position."""
        if pos.side == Side.BACK:
            # Backed at entry, laid at exit
            # Profit = stake * (exit_odds - entry_odds) / entry_odds (approx)
            # More precisely: backed at entry_odds, then lay at exit_odds
            return pos.stake * (exit_price - pos.entry_odds) / pos.entry_odds
        else:
            # Laid at entry, backed at exit
            return pos.stake * (pos.entry_odds - exit_price) / pos.entry_odds

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def reconcile_on_startup(self) -> None:
        """Fetch open orders and reconstruct positions from API state."""
        log.info("Starting position reconciliation")
        try:
            orders_resp = await self.client.list_current_orders()
            open_orders = orders_resp.get("currentOrders", [])

            for order in open_orders:
                market_id = order.get("marketId", "")
                sel_id = order.get("selectionId", 0)
                side_str = order.get("side", "")
                matched = order.get("sizeMatched", 0)
                price = order.get("averagePriceMatched", order.get("price", 0))

                if matched <= 0 or not market_id:
                    # Cancel unmatched orders from previous session
                    bet_id = order.get("betId", "")
                    if bet_id:
                        await self._cancel_order(market_id, bet_id)
                    continue

                side = Side.BACK if side_str == "BACK" else Side.LAY
                pos = Position(
                    market_id=market_id,
                    selection_id=sel_id,
                    side=side,
                    entry_odds=price,
                    stake=round(matched, 2),
                    entry_time=time.time() - 120,  # assume old
                    score_at_entry="reconciled",
                )
                self.positions[market_id] = pos
                log.info(
                    "Reconciled position: %s %s @ %.2f (£%.2f) on %s",
                    side.value, sel_id, price, matched, market_id,
                )

            log.info("Reconciliation complete: %d open positions", len(self.positions))

        except Exception as e:
            log.error("Position reconciliation failed: %s", e)

    async def cancel_all_orders(self, market_id: str) -> None:
        """Cancel all orders in a market (used on suspension)."""
        try:
            await self.client.cancel_orders(market_id)
            log.info("Cancelled all orders on %s", market_id)
        except Exception as e:
            log.error("Failed to cancel orders on %s: %s", market_id, e)

    async def close_all_positions(self) -> list[tuple[str, float]]:
        """Emergency close all positions at market. Returns list of (market_id, net_pnl)."""
        results = []
        for market_id in list(self.positions.keys()):
            pos = self.positions[market_id]
            # Use a very aggressive price to ensure fill
            close_side = "LAY" if pos.side == Side.BACK else "BACK"
            instruction = self.client.build_market_order(
                selection_id=pos.selection_id,
                side=close_side,
                size=pos.stake,
            )
            try:
                result = await self.client.place_orders(market_id, [instruction])
                reports = result.get("instructionReports", [])
                exit_price = pos.entry_odds  # fallback
                if reports:
                    exit_price = reports[0].get("averagePriceMatched", exit_price)
                gross = self._calc_gross_profit(pos, exit_price)
                net = gross - max(0, gross) * self.config.commission_rate
                results.append((market_id, net))
                del self.positions[market_id]
            except Exception as e:
                log.error("Failed to close position on %s: %s", market_id, e)

        self.pending_entries.clear()
        return results

    async def _cancel_order(self, market_id: str, bet_id: str) -> None:
        try:
            await self.client.cancel_orders(
                market_id,
                [{"betId": bet_id}],
            )
        except Exception as e:
            log.warning("Cancel order failed (%s/%s): %s", market_id, bet_id, e)
