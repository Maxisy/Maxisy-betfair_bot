"""Trade logger and daily metrics.

Every trade → trades.jsonl. Daily summary at 00:00 UTC.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ExitReason, Position, ScoreSource, TradeLog

log = logging.getLogger(__name__)

TRADES_FILE = Path("trades.jsonl")


class TradeLogger:
    """Writes every trade to trades.jsonl and computes daily metrics."""

    def __init__(self, paper_mode: bool = True) -> None:
        self.paper_mode = paper_mode

        # Daily counters
        self.daily_trades: list[TradeLog] = []
        self.daily_gross_pnl: float = 0.0
        self.daily_commission: float = 0.0

    def reset_daily(self) -> None:
        self.daily_trades.clear()
        self.daily_gross_pnl = 0.0
        self.daily_commission = 0.0

    async def log_trade(
        self,
        pos: Position,
        exit_odds: float,
        net_profit: float,
        reason: ExitReason,
    ) -> None:
        """Log a completed trade to file and in-memory metrics."""
        now = datetime.now(timezone.utc)

        # Calculate components
        if pos.side.value == "BACK":
            gross = pos.stake * (exit_odds - pos.entry_odds) / pos.entry_odds
        else:
            gross = pos.stake * (pos.entry_odds - exit_odds) / pos.entry_odds

        commission = max(0, gross) * 0.05
        net = gross - commission

        entry = TradeLog(
            timestamp=now.isoformat(),
            trade_id=pos.trade_id,
            market_id=pos.market_id,
            selection_id=pos.selection_id,
            event_name=pos.event_name,
            tournament=pos.tournament,
            surface=pos.surface,
            side=pos.side.value,
            entry_odds=pos.entry_odds,
            exit_odds=exit_odds,
            stake=pos.stake,
            gross_profit=round(gross, 4),
            commission=round(commission, 4),
            net_profit=round(net, 4),
            hold_seconds=round(pos.hold_seconds, 1),
            exit_reason=reason.value,
            model_odds=pos.model_odds_at_entry,
            market_odds=pos.market_odds_at_entry,
            edge_pct=round(pos.edge_at_entry * 100, 2),
            score_at_entry=pos.score_at_entry,
            model_state_age_seconds=round(pos.model_state_age_at_entry, 1),
            score_source=pos.score_source_at_entry.value,
            paper_trade=self.paper_mode,
        )

        # Write to file
        try:
            with open(TRADES_FILE, "a") as f:
                f.write(json.dumps(entry.__dict__) + "\n")
        except Exception as e:
            log.error("Failed to write trade log: %s", e)

        # Update daily counters
        self.daily_trades.append(entry)
        self.daily_gross_pnl += gross
        self.daily_commission += commission

    def daily_summary(self) -> dict[str, Any]:
        """Compute daily review metrics."""
        trades = self.daily_trades
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "commission": 0.0,
                "net_pnl": 0.0,
            }

        wins = sum(1 for t in trades if t.net_profit > 0)
        total = len(trades)

        exit_reasons: dict[str, int] = {}
        total_hold = 0.0
        total_edge = 0.0
        total_age = 0.0
        best = max(trades, key=lambda t: t.net_profit)
        worst = min(trades, key=lambda t: t.net_profit)

        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
            total_hold += t.hold_seconds
            total_edge += t.edge_pct
            total_age += t.model_state_age_seconds

        return {
            "total_trades": total,
            "win_rate": round(wins / total, 4) if total else 0.0,
            "gross_pnl": round(self.daily_gross_pnl, 2),
            "commission": round(self.daily_commission, 2),
            "net_pnl": round(self.daily_gross_pnl - self.daily_commission, 2),
            "avg_hold_seconds": round(total_hold / total, 1),
            "avg_edge_pct": round(total_edge / total, 2),
            "avg_model_age_seconds": round(total_age / total, 1),
            "exit_reasons": exit_reasons,
            "best_trade": round(best.net_profit, 2),
            "worst_trade": round(worst.net_profit, 2),
            "inference_trades": sum(1 for t in trades if t.score_source == "inference"),
        }

    def save_daily_summary(self) -> None:
        """Write daily summary to logs directory."""
        summary = self.daily_summary()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = Path("logs") / f"daily_{date_str}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)
            log.info("Daily summary saved to %s", path)
        except Exception as e:
            log.error("Failed to save daily summary: %s", e)
