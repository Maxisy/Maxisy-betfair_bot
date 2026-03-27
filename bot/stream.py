"""Betfair Streaming API handler.

Maintains a persistent SSL socket to stream-api.betfair.com:443.
Receives real-time odds updates and distributes them via callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from typing import Any, Callable, Coroutine, Optional

from .config import Config
from .models import MarketState, MarketStatus, RunnerState

log = logging.getLogger(__name__)

STREAM_HOST = "stream-api.betfair.com"
STREAM_PORT = 443
HEARTBEAT_TIMEOUT = 10.0
RECONNECT_DELAY = 2.0


class BetfairStream:
    """Async Betfair Streaming API client."""

    def __init__(
        self,
        config: Config,
        session_token: str,
        on_market_update: Callable[[str, MarketState], Coroutine] | None = None,
    ) -> None:
        self.config = config
        self._session_token = session_token
        self._on_market_update = on_market_update

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._request_id = 0
        self._connected = False
        self._running = False
        self._last_message_time: float = 0.0
        self._reconnect_count = 0

        # Market state cache
        self.markets: dict[str, MarketState] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def update_session_token(self, token: str) -> None:
        self._session_token = token

    async def start(self) -> None:
        """Connect and start processing stream messages."""
        self._running = True
        while self._running:
            try:
                await self._connect()
                await self._authenticate()
                await self._subscribe()
                self._connected = True
                self._reconnect_count = 0
                log.info("Betfair stream connected and subscribed")
                await self._read_loop()
            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                log.error(
                    "Stream error (reconnect #%d): %s",
                    self._reconnect_count, e,
                )
                await self._close_connection()
                if self._running:
                    await asyncio.sleep(RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False
        self._connected = False
        await self._close_connection()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_ctx.load_cert_chain(
            certfile=str(self.config.cert_file),
            keyfile=str(self.config.key_file),
        )
        ssl_ctx.load_default_certs()
        self._reader, self._writer = await asyncio.open_connection(
            STREAM_HOST, STREAM_PORT, ssl=ssl_ctx,
        )
        self._last_message_time = time.time()

    async def _close_connection(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            self._writer = None
            self._reader = None

    async def _send(self, msg: dict[str, Any]) -> None:
        self._request_id += 1
        msg["id"] = self._request_id
        data = json.dumps(msg) + "\r\n"
        assert self._writer is not None
        self._writer.write(data.encode())
        await self._writer.drain()

    # ------------------------------------------------------------------
    # Authentication & subscription
    # ------------------------------------------------------------------

    async def _authenticate(self) -> None:
        await self._send({
            "op": "authentication",
            "appKey": self.config.betfair_app_key,
            "session": self._session_token,
        })
        # Read auth response
        assert self._reader is not None
        line = await asyncio.wait_for(self._reader.readline(), timeout=10)
        msg = json.loads(line)
        if msg.get("op") == "connection":
            # Connection message comes first
            line = await asyncio.wait_for(self._reader.readline(), timeout=10)
            msg = json.loads(line)
        if msg.get("statusCode") != "SUCCESS" and msg.get("op") != "status":
            log.debug("Auth response: %s", msg)

    async def _subscribe(self) -> None:
        await self._send({
            "op": "marketSubscription",
            "marketFilter": {
                "eventTypeIds": ["2"],  # Tennis
                "marketTypes": ["MATCH_ODDS"],
                "bettingTypes": ["ODDS"],
            },
            "marketDataFilter": {
                "fields": [
                    "EX_BEST_OFFERS",
                    "EX_TRADED",
                    "EX_LTP",
                ],
                "ladderLevels": 3,
            },
        })

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    async def _read_loop(self) -> None:
        assert self._reader is not None

        while self._running:
            try:
                line = await asyncio.wait_for(
                    self._reader.readline(), timeout=HEARTBEAT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.warning("Stream heartbeat timeout — reconnecting")
                break

            if not line:
                log.warning("Stream EOF — reconnecting")
                break

            self._last_message_time = time.time()

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            op = msg.get("op")
            if op == "mcm":
                await self._handle_market_change(msg)
            elif op == "status":
                if msg.get("statusCode") != "SUCCESS":
                    log.error("Stream status error: %s", msg)
            # heartbeat and connection messages are expected, just update timestamp

    async def _handle_market_change(self, msg: dict[str, Any]) -> None:
        for mc in msg.get("mc", []):
            market_id = mc.get("id", "")
            if not market_id:
                continue

            market = self.markets.get(market_id)
            if market is None:
                market = MarketState(market_id=market_id)
                self.markets[market_id] = market

            # Update market definition
            market_def = mc.get("marketDefinition")
            if market_def:
                status = market_def.get("status", "OPEN")
                market.status = MarketStatus(status) if status in MarketStatus.__members__ else MarketStatus.OPEN
                market.in_play = market_def.get("inPlay", False)
                market.total_matched = market_def.get("totalMatched", market.total_matched)
                event = market_def.get("eventName", "")
                if event:
                    market.event_name = event

            # Update runners
            for rc in mc.get("rc", []):
                sel_id = rc.get("id", 0)
                if sel_id == 0:
                    continue

                runner = market.runners.get(sel_id)
                if runner is None:
                    runner = RunnerState(selection_id=sel_id)
                    market.runners[sel_id] = runner

                # Best available to back
                batb = rc.get("batb", [])
                if batb:
                    # batb is [[level, price, size], ...]
                    best = min(batb, key=lambda x: x[0])  # level 0 is best
                    runner.best_back_price = best[1]
                    runner.best_back_size = best[2]

                # Best available to lay
                batl = rc.get("batl", [])
                if batl:
                    best = min(batl, key=lambda x: x[0])
                    runner.best_lay_price = best[1]
                    runner.best_lay_size = best[2]

                # Last traded price
                ltp = rc.get("ltp")
                if ltp is not None:
                    runner.last_traded_price = ltp

                # Total matched on runner
                tv = rc.get("tv")
                if tv is not None:
                    market.total_matched = max(market.total_matched, tv)

            # Notify callback
            if self._on_market_update:
                try:
                    await self._on_market_update(market_id, market)
                except Exception as e:
                    log.error("Market update callback error for %s: %s", market_id, e)


class PaperBetfairStream(BetfairStream):
    """Stub stream for paper trading — does nothing, markets fed externally."""

    async def start(self) -> None:
        self._running = True
        self._connected = True
        log.info("Paper stream started — waiting for external market data")
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        self._connected = False
