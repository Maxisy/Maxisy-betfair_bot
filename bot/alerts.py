"""Alert system — Discord/Telegram webhook notifications."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp

from .config import Config

log = logging.getLogger(__name__)


class AlertSystem:
    """Sends alerts via Discord or Telegram webhook."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def send(self, level: str, message: str, data: dict[str, Any] | None = None) -> None:
        """Send an alert. Level: info, warning, critical."""
        if not self.config.alert_webhook_url:
            log.info("ALERT [%s]: %s", level, message)
            return

        timestamp = datetime.now(timezone.utc).isoformat()
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "📋")

        # Format for Discord webhook
        payload = {
            "content": f"{emoji} **[{level.upper()}]** {message}",
            "embeds": [],
        }

        if data:
            fields = [{"name": k, "value": str(v), "inline": True} for k, v in data.items()]
            payload["embeds"].append({
                "title": "Details",
                "fields": fields[:25],
                "timestamp": timestamp,
            })

        await self._post(payload)

    async def send_bot_started(self, mode: str) -> None:
        await self.send("info", f"Bot started in **{mode}** mode")

    async def send_stream_disconnected(self, reconnect_count: int) -> None:
        await self.send("warning", f"Stream disconnected (reconnect #{reconnect_count})")

    async def send_stream_reconnected(self, downtime: float) -> None:
        await self.send("info", f"Stream reconnected after {downtime:.0f}s downtime")

    async def send_daily_loss_limit(self, pnl: float) -> None:
        await self.send(
            "critical",
            f"Daily loss limit hit — bot stopped. P&L: £{pnl:.2f}",
        )

    async def send_goalserve_down(self, seconds: float, matches_affected: int) -> None:
        await self.send(
            "warning",
            f"Goalserve down for {seconds:.0f}s — {matches_affected} matches affected",
        )

    async def send_win_rate_alert(self, win_rate: float) -> None:
        await self.send(
            "warning",
            f"Win rate dropped to {win_rate:.1%} — stakes reduced to 50%",
        )

    async def send_daily_summary(self, summary: dict[str, Any]) -> None:
        msg = (
            f"Daily Summary: {summary['total_trades']} trades, "
            f"{summary['win_rate']:.1%} win rate, "
            f"net £{summary['net_pnl']:.2f}"
        )
        await self.send("info", msg, summary)

    async def _post(self, payload: dict) -> None:
        if not self._session:
            return
        try:
            async with self._session.post(
                self.config.alert_webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    log.warning("Alert webhook returned %d", resp.status)
        except Exception as e:
            log.error("Alert send failed: %s", e)
