"""Betfair APING REST client with cert-based authentication."""

from __future__ import annotations

import logging
import ssl
import time
from typing import Any, Optional

import aiohttp

from .config import Config

log = logging.getLogger(__name__)

CERT_LOGIN_URL = "https://identitysso-cert.betfair.com/api/certlogin"
APING_URL = "https://api.betfair.com/exchange"
SPORTS_URL = f"{APING_URL}/betting/rest/v1.0"
ACCOUNT_URL = f"{APING_URL}/account/rest/v1.0"


class BetfairClient:
    """Async Betfair APING client."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._session_token: str = ""
        self._session_expires: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        await self.login()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def login(self) -> str:
        """Cert-based login. Returns session token."""
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.load_cert_chain(
            certfile=str(self.config.cert_file),
            keyfile=str(self.config.key_file),
        )
        connector = aiohttp.TCPConnector(ssl=ssl_ctx)
        async with aiohttp.ClientSession(connector=connector) as sess:
            resp = await sess.post(
                CERT_LOGIN_URL,
                data={
                    "username": self.config.betfair_username,
                    "password": self.config.betfair_password,
                },
                headers={"X-Application": self.config.betfair_app_key},
            )
            body = await resp.json()

        if body.get("loginStatus") != "SUCCESS":
            raise RuntimeError(f"Betfair login failed: {body}")

        self._session_token = body["sessionToken"]
        self._session_expires = time.time() + 3 * 3600  # refresh before 4h
        log.info("Betfair authenticated successfully")
        return self._session_token

    @property
    def session_token(self) -> str:
        return self._session_token

    async def _ensure_session(self) -> None:
        if not self._session_token or time.time() > self._session_expires:
            await self.login()

    def _headers(self) -> dict[str, str]:
        return {
            "X-Application": self.config.betfair_app_key,
            "X-Authentication": self._session_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Generic APING call with retry
    # ------------------------------------------------------------------

    async def _call(self, url: str, method: str, params: dict[str, Any],
                    retries: int = 3) -> Any:
        await self._ensure_session()
        assert self._session is not None

        full_url = f"{url}/{method}/"
        backoff = 1.0

        for attempt in range(retries):
            try:
                async with self._session.post(
                    full_url,
                    json={"filter": params} if "filter" in method.lower() else params,
                    headers=self._headers(),
                ) as resp:
                    # Handle auth errors
                    if resp.status == 400:
                        body = await resp.json()
                        error = body.get("detail", {}).get("APINGException", {})
                        error_code = error.get("errorCode", "")
                        if error_code in ("NO_SESSION", "INVALID_SESSION_INFORMATION"):
                            log.warning("Session expired, re-authenticating")
                            await self.login()
                            continue
                        raise RuntimeError(f"APING error: {body}")

                    resp.raise_for_status()
                    return await resp.json()

            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt < retries - 1:
                    log.warning("APING call %s failed (attempt %d): %s", method, attempt + 1, e)
                    await _sleep(backoff)
                    backoff *= 2
                else:
                    raise

        return None  # unreachable

    # ------------------------------------------------------------------
    # Betting API methods
    # ------------------------------------------------------------------

    async def list_market_catalogue(
        self,
        event_type_ids: list[str] | None = None,
        market_type_codes: list[str] | None = None,
        in_play_only: bool = False,
        max_results: int = 200,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "filter": {
                "eventTypeIds": event_type_ids or ["2"],  # Tennis
                "marketTypeCodes": market_type_codes or ["MATCH_ODDS"],
            },
            "maxResults": max_results,
            "marketProjection": [
                "EVENT",
                "RUNNER_DESCRIPTION",
                "MARKET_START_TIME",
                "COMPETITION",
            ],
        }
        if in_play_only:
            params["filter"]["inPlayOnly"] = True
        return await self._call(SPORTS_URL, "listMarketCatalogue", params)

    async def list_market_book(
        self,
        market_ids: list[str],
        price_projection: dict | None = None,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "marketIds": market_ids,
            "priceProjection": price_projection or {
                "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
            },
        }
        return await self._call(SPORTS_URL, "listMarketBook", params)

    async def place_orders(
        self,
        market_id: str,
        instructions: list[dict],
    ) -> dict:
        params = {
            "marketId": market_id,
            "instructions": instructions,
        }
        return await self._call(SPORTS_URL, "placeOrders", params)

    async def cancel_orders(
        self,
        market_id: str,
        instructions: list[dict] | None = None,
    ) -> dict:
        params: dict[str, Any] = {"marketId": market_id}
        if instructions:
            params["instructions"] = instructions
        return await self._call(SPORTS_URL, "cancelOrders", params)

    async def replace_orders(
        self,
        market_id: str,
        instructions: list[dict],
    ) -> dict:
        params = {
            "marketId": market_id,
            "instructions": instructions,
        }
        return await self._call(SPORTS_URL, "replaceOrders", params)

    async def list_current_orders(self) -> dict:
        return await self._call(SPORTS_URL, "listCurrentOrders", {})

    async def list_cleared_orders(
        self,
        bet_status: str = "SETTLED",
    ) -> dict:
        params = {"betStatus": bet_status}
        return await self._call(SPORTS_URL, "listClearedOrders", params)

    async def get_account_funds(self) -> dict:
        return await self._call(ACCOUNT_URL, "getAccountFunds", {})

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def build_limit_order(
        self,
        selection_id: int,
        side: str,
        price: float,
        size: float,
    ) -> dict:
        """Build a LIMIT order instruction with LAPSE persistence."""
        return {
            "selectionId": selection_id,
            "side": side,
            "orderType": "LIMIT",
            "limitOrder": {
                "size": round(size, 2),
                "price": price,
                "persistenceType": "LAPSE",
            },
        }

    def build_market_order(
        self,
        selection_id: int,
        side: str,
        size: float,
    ) -> dict:
        """Build a LIMIT order at aggressive price to simulate market order.

        Betfair doesn't have true market orders. We use the worst acceptable
        price to get immediate fill.
        """
        # For BACK, use highest price; for LAY, use lowest price
        price = 1000.0 if side == "BACK" else 1.01
        return {
            "selectionId": selection_id,
            "side": side,
            "orderType": "LIMIT",
            "limitOrder": {
                "size": round(size, 2),
                "price": price,
                "persistenceType": "LAPSE",
            },
        }


class PaperBetfairClient(BetfairClient):
    """Simulated client for paper trading — no real orders placed."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._order_counter = 0

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._session_token = "PAPER_SESSION"
        log.info("Paper trading mode — no real Betfair authentication")

    async def login(self) -> str:
        self._session_token = "PAPER_SESSION"
        return self._session_token

    async def place_orders(self, market_id: str, instructions: list[dict]) -> dict:
        self._order_counter += 1
        results = []
        for inst in instructions:
            results.append({
                "status": "SUCCESS",
                "instruction": inst,
                "betId": f"PAPER_{self._order_counter}",
                "placedDate": "",
                "averagePriceMatched": inst["limitOrder"]["price"],
                "sizeMatched": inst["limitOrder"]["size"],
                "orderStatus": "EXECUTABLE",
            })
        log.info("PAPER: placed %d orders on %s", len(instructions), market_id)
        return {"status": "SUCCESS", "instructionReports": results}

    async def cancel_orders(self, market_id: str, instructions: list[dict] | None = None) -> dict:
        log.info("PAPER: cancelled orders on %s", market_id)
        return {"status": "SUCCESS", "instructionReports": []}

    async def replace_orders(self, market_id: str, instructions: list[dict]) -> dict:
        return await self.place_orders(market_id, instructions)

    async def list_current_orders(self) -> dict:
        return {"currentOrders": []}

    async def list_cleared_orders(self, bet_status: str = "SETTLED") -> dict:
        return {"clearedOrders": []}

    async def get_account_funds(self) -> dict:
        return {"availableToBetBalance": 1000.0}

    async def list_market_catalogue(self, **kwargs) -> list[dict]:
        return []

    async def list_market_book(self, market_ids: list[str], **kwargs) -> list[dict]:
        return []


async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
