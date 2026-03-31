"""Tennis Abstract serve stats loader.

Fetches serve win percentages per player per surface from tennisabstract.com.

Data sources:
  - Player list: /jsplayers/curr_rank_atp.js and curr_rank_wta.js
  - Match data:  /cgi-bin/player-classic.cgi?p={FirstLast}
    Contains inline `var matchmx` JS array with per-match serve stats.
    Fields at indices: [2]=surface, [29]=pts, [30]=firsts, [31]=fwon, [32]=swon
    Service points won % = (fwon + swon) / pts

Loads on startup and refreshes weekly. Falls back to surface defaults.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import aiohttp

from .config import Config

log = logging.getLogger(__name__)

STATS_FILE = Path("data/serve_stats.json")
REFRESH_INTERVAL = 7 * 24 * 3600  # 1 week

TA_BASE = "https://www.tennisabstract.com"
ATP_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_atp.js"
WTA_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_wta.js"
# ATP: inline matchmx on player-classic.cgi
ATP_PLAYER_URL = f"{TA_BASE}/cgi-bin/player-classic.cgi?p={{name}}"
# WTA: matchmx in external /jsmatches/{{Name}}.js
WTA_MATCHES_URL = f"{TA_BASE}/jsmatches/{{name}}.js"

# matchmx field indices (from var matchhead on player-classic.cgi)
# [0]date [1]tourn [2]surf ... [23]pts [24]firsts [25]fwon [26]swon
IDX_SURFACE = 2
IDX_PTS = 23
IDX_FWON = 25
IDX_SWON = 26


class ServeStatsLoader:
    """Load and cache serve win percentages per player per surface."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # "firstname lastname" (lower) -> {"hard": 0.63, "clay": 0.60, ...}
        self.stats: dict[str, dict[str, float]] = {}
        self._last_refresh: float = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def start(self) -> None:
        """Load cached stats, then start weekly refresh loop."""
        self._load_cached()
        self._session = aiohttp.ClientSession()
        self._running = True

        # Try initial scrape if cache is stale
        if time.time() - self._last_refresh > REFRESH_INTERVAL:
            await self._refresh()

        # Background refresh loop
        while self._running:
            await asyncio.sleep(3600)  # check hourly
            if time.time() - self._last_refresh > REFRESH_INTERVAL:
                await self._refresh()

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    def get_serve_pct(self, player_name: str, surface: str) -> float:
        """Get serve win % for a player on a surface.

        Tries full name match, then surname-only fallback.
        Falls back to surface defaults if player not found.
        """
        from unidecode import unidecode
        name_clean = unidecode(player_name).lower().strip()

        # Try full name
        if name_clean in self.stats and surface in self.stats[name_clean]:
            return self.stats[name_clean][surface]

        # Try surname only
        surname = name_clean.split()[-1] if name_clean else ""
        for key, val in self.stats.items():
            if key.endswith(surname) and surface in val:
                return val[surface]

        return self.config.surface_defaults.get(surface, 0.63)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cached(self) -> None:
        if STATS_FILE.exists():
            try:
                data = json.loads(STATS_FILE.read_text())
                self.stats = data.get("stats", {})
                self._last_refresh = data.get("last_refresh", 0)
                log.info("Loaded %d player serve stats from cache", len(self.stats))
            except Exception as e:
                log.warning("Failed to load cached stats: %s", e)

    def _save_cache(self) -> None:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats,
            "last_refresh": self._last_refresh,
        }
        STATS_FILE.write_text(json.dumps(data, indent=2))
        log.info("Saved %d player serve stats to cache", len(self.stats))

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        """Scrape Tennis Abstract for serve stats."""
        log.info("Refreshing serve stats from Tennis Abstract")
        assert self._session is not None

        try:
            atp_names = await self._fetch_player_list(ATP_RANKS_URL)
            log.info("Found %d ATP players", len(atp_names))
            await self._scrape_players(atp_names, tour="atp")

            wta_names = await self._fetch_player_list(WTA_RANKS_URL)
            log.info("Found %d WTA players", len(wta_names))
            await self._scrape_players(wta_names, tour="wta")

            self._last_refresh = time.time()
            self._save_cache()
            log.info("Serve stats refresh complete: %d players", len(self.stats))
        except Exception as e:
            log.error("Serve stats refresh failed: %s", e)

    async def _fetch_player_list(self, url: str) -> list[str]:
        """Fetch player name list from Tennis Abstract JS file.

        File contains: var currRank = {"FirstName LastName": "rank", ...}
        """
        assert self._session is not None
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    log.warning("Player list fetch failed: %d from %s", resp.status, url)
                    return []
                text = await resp.text()
        except Exception as e:
            log.warning("Player list fetch error: %s", e)
            return []

        # Parse: var currRank = {"Name": "rank", ...}
        match = re.search(r'var\s+currRank\s*=\s*(\{[^}]+\})', text)
        if not match:
            log.warning("Could not parse player list from %s", url)
            return []

        try:
            data = json.loads(match.group(1))
            return list(data.keys())
        except json.JSONDecodeError:
            log.warning("JSON parse error for player list")
            return []

    async def _scrape_players(self, names: list[str], tour: str) -> None:
        """Scrape serve stats for a list of players."""
        scraped = 0
        for name in names:
            url_name = name.replace(" ", "")
            try:
                stats = await self._fetch_matchmx(url_name, tour)
                if stats:
                    self.stats[name.lower()] = stats
                    scraped += 1
            except Exception as e:
                log.debug("Failed to scrape %s: %s", name, e)

            await asyncio.sleep(0.3)  # be polite

        log.info("Scraped serve stats for %d/%d players", scraped, len(names))

    async def _fetch_matchmx(
        self, url_name: str, tour: str,
    ) -> dict[str, float] | None:
        """Fetch matchmx data for a player and compute serve stats.

        ATP: inline var matchmx in player-classic.cgi
        WTA: var matchmx in /jsmatches/{Name}.js
        """
        assert self._session is not None

        if tour == "atp":
            url = ATP_PLAYER_URL.format(name=url_name)
        else:
            url = WTA_MATCHES_URL.format(name=url_name)

        async with self._session.get(
            url, timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()

        # Extract matchmx array
        match = re.search(r'var\s+matchmx\s*=\s*(\[.+?\])\s*;', text, re.DOTALL)
        if not match:
            return None

        try:
            import ast
            matches = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            return None

        # Aggregate serve points won by surface
        # Only use recent matches (2024+)
        surface_pts: dict[str, list[int]] = {}  # surface -> [won, total]

        for row in matches:
            if len(row) <= IDX_SWON + 1:
                continue

            date_str = str(row[0])
            # Only recent data
            if date_str < "2024":
                continue

            surface = str(row[IDX_SURFACE]).lower().strip()
            if surface not in ("hard", "clay", "grass"):
                continue

            try:
                pts = int(row[IDX_PTS]) if row[IDX_PTS] else 0
                fwon = int(row[IDX_FWON]) if row[IDX_FWON] else 0
                swon = int(row[IDX_SWON]) if row[IDX_SWON] else 0
            except (ValueError, TypeError):
                continue

            if pts < 20:  # skip walkovers/retirements
                continue

            won = fwon + swon
            if surface not in surface_pts:
                surface_pts[surface] = [0, 0]
            surface_pts[surface][0] += won
            surface_pts[surface][1] += pts

        if not surface_pts:
            return None

        result: dict[str, float] = {}
        for surface, (won, total) in surface_pts.items():
            if total >= 100:  # need enough data
                pct = won / total
                if 0.30 < pct < 0.90:  # sanity check
                    result[surface] = round(pct, 4)

        return result if result else None
