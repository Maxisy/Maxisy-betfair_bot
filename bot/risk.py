"""Risk manager — enforces all trading limits.

Approves or rejects every trade signal. Triggers kill switch on fatal conditions.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Coroutine, Optional

from .config import Config
from .models import Position, ScoreSource

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(
        self,
        config: Config,
        on_kill_switch: Optional[Callable[[], Coroutine]] = None,
    ) -> None:
        self.config = config
        self._on_kill_switch = on_kill_switch

        # Tracking state
        self.daily_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.float_balance: float = 1000.0  # updated from account balance
        self.recent_trades: deque[float] = deque()  # timestamps of recent trades
        self.recent_results: deque[bool] = deque(maxlen=50)  # win/loss of last 50
        self.kill_switch_active: bool = False

        # Rejection counters
        self.rejections: dict[str, int] = {}

    def reset_daily(self) -> None:
        self.daily_pnl = 0.0
        self.daily_trade_count = 0
        self.rejections.clear()

    @property
    def win_rate(self) -> float:
        if not self.recent_results:
            return 1.0
        return sum(1 for w in self.recent_results if w) / len(self.recent_results)

    @property
    def is_stake_reduced(self) -> bool:
        """Check if stakes should be halved due to daily loss or low win rate."""
        if self.daily_pnl <= -self.config.daily_loss_half_stake:
            return True
        if len(self.recent_results) >= 50 and self.win_rate < 0.40:
            return True
        return False

    @property
    def is_win_rate_recovered(self) -> bool:
        return self.win_rate >= 0.45

    def calculate_stake(self, is_inference: bool = False) -> float:
        """Calculate stake for next trade."""
        base = self.float_balance * self.config.stake_pct
        stake = min(base, self.config.phase_max_stake)

        if self.is_stake_reduced:
            stake *= 0.5
        if is_inference:
            stake *= self.config.goalserve_fallback_stake_pct

        return round(max(0.01, stake), 2)

    def check_trade(
        self,
        stake: float,
        market_exposure: float,
        portfolio_exposure: float,
        is_inference: bool = False,
    ) -> tuple[bool, str]:
        """Approve or reject a trade. Returns (approved, reason)."""
        if self.kill_switch_active:
            return self._reject("kill_switch_active")

        # Daily loss limit
        if self.daily_pnl <= -self.config.daily_loss_limit:
            return self._reject("daily_loss_limit")

        # Stake within phase maximum
        if stake > self.config.phase_max_stake:
            return self._reject("stake_exceeds_phase_max")

        # Market exposure
        if market_exposure + stake > self.config.max_market_exposure:
            return self._reject("market_exposure_exceeded")

        # Portfolio exposure
        if portfolio_exposure + stake > self.config.max_portfolio_exposure:
            return self._reject("portfolio_exposure_exceeded")

        # Trade rate limit (20 per minute)
        now = time.time()
        while self.recent_trades and self.recent_trades[0] < now - 60:
            self.recent_trades.popleft()
        if len(self.recent_trades) >= self.config.max_trades_per_minute:
            return self._reject("trade_rate_limit")

        return True, ""

    def record_trade(self, net_profit: float, won: bool) -> None:
        """Record a completed trade for tracking."""
        self.daily_pnl += net_profit
        self.daily_trade_count += 1
        self.recent_trades.append(time.time())
        self.recent_results.append(won)

    async def trigger_kill_switch(self, reason: str) -> None:
        """Activate kill switch — stop all trading."""
        log.critical("KILL SWITCH TRIGGERED: %s", reason)
        self.kill_switch_active = True
        if self._on_kill_switch:
            await self._on_kill_switch()

    def _reject(self, reason: str) -> tuple[bool, str]:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        log.debug("Risk rejected: %s", reason)
        return False, reason
