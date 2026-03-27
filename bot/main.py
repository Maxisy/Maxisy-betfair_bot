"""Main application entry point — asyncio orchestrator.

Startup sequence:
1. Load config and initialise components
2. Authenticate with Betfair
3. Reconcile open positions
4. Start Betfair stream, Goalserve poller, serve stats loader
5. Run match mapper periodically
6. Run trading engine on every stream update
7. Run end-of-day cycle at 00:00 UTC
8. Graceful shutdown on signal
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from .alerts import AlertSystem
from .betfair_client import BetfairClient, PaperBetfairClient
from .config import Config
from .goalserve import GoalservePoller
from .logger import TradeLogger
from .market_filter import MarketFilter
from .matcher import map_matches_to_markets
from .models import ExitReason, MarketState, Position, ScoreState
from .positions import PositionTracker
from .risk import RiskManager
from .serve_stats import ServeStatsLoader
from .stream import BetfairStream, PaperBetfairStream
from .trading import TradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")


class Bot:
    """Main bot orchestrator."""

    def __init__(self) -> None:
        self.config = Config()
        self._shutdown_event = asyncio.Event()

        # Components — initialised in start()
        self.client: BetfairClient
        self.stream: BetfairStream
        self.goalserve: GoalservePoller
        self.serve_stats: ServeStatsLoader
        self.market_filter: MarketFilter
        self.risk: RiskManager
        self.positions: PositionTracker
        self.engine: TradingEngine
        self.trade_logger: TradeLogger
        self.alerts: AlertSystem

    async def start(self) -> None:
        log.info("Starting bot in %s mode", self.config.env)

        # --- Initialise components ---
        if self.config.is_paper:
            self.client = PaperBetfairClient(self.config)
        else:
            self.client = BetfairClient(self.config)

        self.alerts = AlertSystem(self.config)
        self.market_filter = MarketFilter(self.config)
        self.risk = RiskManager(
            self.config,
            on_kill_switch=self._kill_switch,
        )
        self.positions = PositionTracker(self.config, self.client)
        self.engine = TradingEngine(
            self.config, self.positions, self.risk, self.market_filter,
        )
        self.trade_logger = TradeLogger(paper_mode=self.config.is_paper)
        self.serve_stats = ServeStatsLoader(self.config)
        self.goalserve = GoalservePoller(
            self.config,
            on_scores_updated=self._on_scores_updated,
        )

        # Wire up callbacks
        self.engine.on_trade_closed = self._on_trade_closed
        self.engine.on_alert = self._on_alert

        # --- Start services ---
        await self.client.start()
        await self.alerts.start()
        await self.alerts.send_bot_started(self.config.env)

        # Reconcile positions before trading
        await self.positions.reconcile_on_startup()

        # Update float balance
        try:
            funds = await self.client.get_account_funds()
            self.risk.float_balance = funds.get("availableToBetBalance", 1000.0)
            log.info("Account balance: £%.2f", self.risk.float_balance)
        except Exception as e:
            log.warning("Could not fetch account balance: %s", e)

        # Create stream
        if self.config.is_paper:
            self.stream = PaperBetfairStream(
                self.config,
                self.client.session_token,
                on_market_update=self._on_market_update,
            )
        else:
            self.stream = BetfairStream(
                self.config,
                self.client.session_token,
                on_market_update=self._on_market_update,
            )

        # Launch concurrent tasks
        tasks = [
            asyncio.create_task(self.stream.start(), name="stream"),
            asyncio.create_task(self.goalserve.start(), name="goalserve"),
            asyncio.create_task(self.serve_stats.start(), name="serve_stats"),
            asyncio.create_task(self._match_mapper_loop(), name="matcher"),
            asyncio.create_task(self._position_monitor_loop(), name="pos_monitor"),
            asyncio.create_task(self._eod_loop(), name="eod"),
            asyncio.create_task(self._goalserve_health_loop(), name="gs_health"),
        ]

        log.info("All components started — trading active")

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cancel all tasks
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Cleanup
        await self._shutdown()

    async def _shutdown(self) -> None:
        log.info("Shutting down...")
        # Close all positions
        if self.positions.positions:
            results = await self.positions.close_all_positions()
            for mid, pnl in results:
                log.info("Shutdown close: %s → £%.2f", mid, pnl)

        await self.stream.stop()
        await self.goalserve.stop()
        await self.serve_stats.stop()
        await self.client.close()
        await self.alerts.stop()
        log.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    async def _on_market_update(self, market_id: str, market: MarketState) -> None:
        """Called by stream on every odds update."""
        await self.engine.on_market_update(
            market_id, market, self.goalserve.scores,
        )

    async def _on_scores_updated(self) -> None:
        """Called after Goalserve poll updates scores."""
        # Update serve percentages from loaded stats
        for state in self.goalserve.scores.values():
            state.player1_serve_pct = self.serve_stats.get_serve_pct(
                state.player1_name, state.surface,
            )
            state.player2_serve_pct = self.serve_stats.get_serve_pct(
                state.player2_name, state.surface,
            )

    async def _on_trade_closed(
        self,
        pos: Position,
        exit_odds: float,
        net_profit: float,
        reason: ExitReason,
    ) -> None:
        await self.trade_logger.log_trade(pos, exit_odds, net_profit, reason)

    async def _on_alert(self, level: str, message: str) -> None:
        await self.alerts.send(level, message)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _match_mapper_loop(self) -> None:
        """Periodically map Goalserve matches to Betfair markets."""
        while not self._shutdown_event.is_set():
            try:
                unmapped = {
                    mid: s for mid, s in self.goalserve.scores.items()
                    if not s.betfair_market_id
                }
                if unmapped:
                    catalogues = await self.client.list_market_catalogue(
                        in_play_only=True,
                    )
                    map_matches_to_markets(unmapped, catalogues)
            except Exception as e:
                log.error("Match mapper error: %s", e)

            await asyncio.sleep(30)

    async def _position_monitor_loop(self) -> None:
        """Check exit conditions every second for all open positions."""
        while not self._shutdown_event.is_set():
            for market_id in list(self.positions.positions.keys()):
                market = self.stream.markets.get(market_id)
                if market is None:
                    continue
                score = self.engine._find_score(market_id, self.goalserve.scores)
                try:
                    await self.engine._check_exit(market_id, market, score)
                except Exception as e:
                    log.error("Position monitor error for %s: %s", market_id, e)

            await asyncio.sleep(1)

    async def _eod_loop(self) -> None:
        """End-of-day cycle at 00:00 UTC."""
        while not self._shutdown_event.is_set():
            now = datetime.now(timezone.utc)
            # Calculate seconds until next 00:00 UTC
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now.hour == 0 and now.minute == 0:
                # It's midnight — run EOD
                await self._run_eod()
                await asyncio.sleep(60)  # avoid double-trigger
            else:
                # Sleep until next midnight
                from datetime import timedelta
                if tomorrow <= now:
                    tomorrow += timedelta(days=1)
                wait = (tomorrow - now).total_seconds()
                await asyncio.sleep(min(wait, 3600))  # check at least hourly

    async def _run_eod(self) -> None:
        """End-of-day processing."""
        log.info("Running end-of-day cycle")
        self.trade_logger.save_daily_summary()
        summary = self.trade_logger.daily_summary()
        await self.alerts.send_daily_summary(summary)

        # Reset counters
        self.trade_logger.reset_daily()
        self.risk.reset_daily()
        self.market_filter.reset_daily()
        log.info("End-of-day complete — counters reset")

    async def _goalserve_health_loop(self) -> None:
        """Monitor Goalserve health and send alerts."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(10)

            if self.goalserve.last_success == 0:
                continue

            down_seconds = self.goalserve.seconds_since_success

            if down_seconds > self.config.goalserve_alert_seconds:
                await self.alerts.send_goalserve_down(
                    down_seconds, len(self.goalserve.scores),
                )

            if down_seconds > self.config.goalserve_pause_seconds:
                # Switch all matches to inference mode
                for state in self.goalserve.scores.values():
                    from .models import ScoreSource
                    state.source = ScoreSource.INFERENCE

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    async def _kill_switch(self) -> None:
        """Emergency shutdown — close everything."""
        log.critical("KILL SWITCH — closing all positions and stopping")
        results = await self.positions.close_all_positions()
        for mid, pnl in results:
            log.info("Kill switch close: %s → £%.2f", mid, pnl)

        await self.alerts.send_daily_loss_limit(self.risk.daily_pnl)
        self._shutdown_event.set()


def main() -> None:
    """CLI entry point."""
    bot = Bot()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle shutdown signals
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: bot._shutdown_event.set())

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
