"""Tennis Abstract serve stats loader.

Scrapes serve win percentages per player per surface from tennisabstract.com.
Loads on startup and refreshes weekly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from .config import Config

log = logging.getLogger(__name__)

STATS_FILE = Path("data/serve_stats.json")
REFRESH_INTERVAL = 7 * 24 * 3600  # 1 week


class ServeStatsLoader:
    """Load and cache serve win percentages per player per surface."""

    def __init__(self, config: Config) -> None:
        self.config = config
        # player_surname_lower -> {"hard": 0.63, "clay": 0.60, "grass": 0.65}
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

        Falls back to surface defaults if player not found.
        """
        from unidecode import unidecode
        surname = unidecode(player_name).lower().strip().split()[-1]

        player_stats = self.stats.get(surname)
        if player_stats and surface in player_stats:
            return player_stats[surface]

        # Surface defaults
        return self.config.surface_defaults.get(surface, 0.63)

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

    async def _refresh(self) -> None:
        """Scrape Tennis Abstract for serve stats."""
        log.info("Refreshing serve stats from Tennis Abstract")
        assert self._session is not None

        try:
            # Scrape ATP players
            await self._scrape_tour("atp")
            # Scrape WTA players
            await self._scrape_tour("wta")

            self._last_refresh = time.time()
            self._save_cache()
            log.info("Serve stats refresh complete: %d players", len(self.stats))
        except Exception as e:
            log.error("Serve stats refresh failed: %s", e)

    async def _scrape_tour(self, tour: str) -> None:
        """Scrape serve stats for ATP or WTA tour."""
        assert self._session is not None

        # Tennis Abstract player list page
        url = f"https://www.tennisabstract.com/cgi-bin/leaders.cgi?f={tour}Main"
        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    log.warning("Tennis Abstract returned %d for %s", resp.status, tour)
                    return
                html = await resp.text()
        except Exception as e:
            log.warning("Failed to fetch Tennis Abstract %s: %s", tour, e)
            return

        soup = BeautifulSoup(html, "html.parser")

        # Find player links and try to scrape individual stats
        # This is a simplified approach — Tennis Abstract's format may vary
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/cgi-bin/player.cgi?" not in href:
                continue

            player_name = link.get_text(strip=True)
            if not player_name:
                continue

            from unidecode import unidecode
            surname = unidecode(player_name).lower().strip().split()[-1]

            # Try to get player-specific serve stats
            try:
                await self._scrape_player(surname, href, tour)
            except Exception:
                pass  # Individual player failures are non-critical

            await asyncio.sleep(0.5)  # Be polite to the server

    async def _scrape_player(self, surname: str, href: str, tour: str) -> None:
        """Scrape individual player serve stats by surface."""
        assert self._session is not None

        base = "https://www.tennisabstract.com"
        url = f"{base}{href}" if href.startswith("/") else href

        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return
                html = await resp.text()
        except Exception:
            return

        soup = BeautifulSoup(html, "html.parser")

        # Look for serve percentages in the page
        # Tennis Abstract typically shows service points won %
        stats: dict[str, float] = {}

        # Parse tables looking for serve stats
        for table in soup.find_all("table"):
            text = table.get_text()
            if "serve" in text.lower() or "service" in text.lower():
                # Try to extract percentages per surface
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        for surface in ("hard", "clay", "grass"):
                            if surface in label:
                                try:
                                    pct_text = cells[-1].get_text(strip=True).replace("%", "")
                                    pct = float(pct_text) / 100.0
                                    if 0.3 < pct < 0.9:  # sanity check
                                        stats[surface] = pct
                                except (ValueError, IndexError):
                                    pass

        if stats:
            if surname not in self.stats:
                self.stats[surname] = {}
            self.stats[surname].update(stats)
