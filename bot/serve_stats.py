"""Tennis Abstract serve stats loader.

Fetches serve win percentages per player per surface from tennisabstract.com.

Data sources:
  - Player list: /jsplayers/curr_rank_atp.js and curr_rank_wta.js
  - Match data (ATP): /cgi-bin/player-classic.cgi?p={FirstLast} (inline matchmx)
  - Match data (WTA): /jsmatches/{FirstLast}.js (external matchmx)
    Fields at indices: [2]=surface, [23]=pts, [25]=fwon, [26]=swon
    Service points won % = (fwon + swon) / pts

Player names in Goalserve are often truncated (e.g. "Daniel Merida" vs
"Daniel Merida Aguilar" on TA). We build a name index keyed by every word
in the name and fuzzy-match Goalserve names against it.

Loads on startup and refreshes weekly. Falls back to surface defaults.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import aiohttp
from unidecode import unidecode

from .config import Config

log = logging.getLogger(__name__)

STATS_FILE = Path("data/serve_stats.json")
REFRESH_INTERVAL = 7 * 24 * 3600  # 1 week

TA_BASE = "https://www.tennisabstract.com"
ATP_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_atp.js"
WTA_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_wta.js"
ATP_PLAYER_URL = f"{TA_BASE}/cgi-bin/player-classic.cgi?p={{name}}"
WTA_MATCHES_URL = f"{TA_BASE}/jsmatches/{{name}}.js"

# matchmx field indices
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
        # Name index: name_word (lower) -> [(full_ta_name, tour)]
        # Keyed by EVERY word in the name for fuzzy matching
        self._name_index: dict[str, list[tuple[str, str]]] = {}

    async def start(self) -> None:
        """Load cached stats, then start weekly refresh loop."""
        self._load_cached()
        self._session = aiohttp.ClientSession()
        self._running = True

        if time.time() - self._last_refresh > REFRESH_INTERVAL:
            await self._refresh()

        while self._running:
            await asyncio.sleep(3600)
            if time.time() - self._last_refresh > REFRESH_INTERVAL:
                await self._refresh()

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    def get_serve_pct(self, player_name: str, surface: str) -> float:
        """Get serve win % for a player on a surface.

        Match strategy:
        1. Exact full name
        2. Goalserve name is prefix of TA name (truncated surnames)
        3. Surname + first-name match
        Falls back to surface defaults if not found.
        """
        name_clean = unidecode(player_name).lower().strip()

        # 1. Exact full name
        if name_clean in self.stats and surface in self.stats[name_clean]:
            return self.stats[name_clean][surface]

        # 2. Goalserve name is prefix of TA name
        for key, val in self.stats.items():
            if key.startswith(name_clean) and surface in val:
                return val[surface]

        # 3. Last-word surname + first name
        parts = name_clean.split()
        if parts:
            surname = parts[-1]
            first = parts[0]
            # Prefer first+surname match
            for key, val in self.stats.items():
                kparts = key.split()
                if kparts and kparts[-1] == surname and kparts[0] == first and surface in val:
                    return val[surface]
            # Just surname
            for key, val in self.stats.items():
                kparts = key.split()
                if kparts and kparts[-1] == surname and surface in val:
                    return val[surface]

        return self.config.surface_defaults.get(surface, 0.58)

    def resolve_ta_name(self, goalserve_name: str) -> tuple[str, str] | None:
        """Resolve a Goalserve player name to (full_ta_name, tour).

        Searches the name index by every word in the Goalserve name,
        then picks the best candidate.
        """
        clean = unidecode(goalserve_name).lower().strip()
        words = clean.split()
        if not words:
            return None

        # Gather candidates from all words in the name
        candidates: dict[str, str] = {}  # ta_name_lower -> tour
        for word in words:
            for ta_name, tour in self._name_index.get(word, []):
                candidates[ta_name.lower()] = tour

        if not candidates:
            return None

        # Score candidates: how many words from goalserve name appear in TA name
        best = None
        best_score = -1
        for ta_lower, tour in candidates.items():
            # Exact match
            if ta_lower == clean:
                return (ta_lower, tour)
            # Prefix match (Goalserve truncated)
            if ta_lower.startswith(clean):
                score = 100 + len(clean)
            else:
                score = sum(1 for w in words if w in ta_lower.split())
            if score > best_score:
                best_score = score
                best = (ta_lower, tour)

        # Require at least 2 matching words (first + last) or prefix match
        if best and best_score >= 2:
            # Find original casing
            for word in words:
                for ta_name, tour in self._name_index.get(word, []):
                    if ta_name.lower() == best[0]:
                        return (ta_name, tour)
            return best

        return None

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cached(self) -> None:
        if STATS_FILE.exists():
            try:
                data = json.loads(STATS_FILE.read_text())
                self.stats = data.get("stats", {})
                self._last_refresh = data.get("last_refresh", 0)
                # Rebuild name index from stored list
                stored_index = data.get("name_index", {})
                if stored_index:
                    self._name_index = {
                        k: [(n, t) for n, t in v]
                        for k, v in stored_index.items()
                    }
                log.info("Loaded %d player serve stats from cache", len(self.stats))
            except Exception as e:
                log.warning("Failed to load cached stats: %s", e)

    def _save_cache(self) -> None:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats,
            "last_refresh": self._last_refresh,
            "name_index": {
                k: [[n, t] for n, t in v]
                for k, v in self._name_index.items()
            },
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

            wta_names = await self._fetch_player_list(WTA_RANKS_URL)
            log.info("Found %d WTA players", len(wta_names))

            # Build name index keyed by every word in name
            self._build_name_index(atp_names, wta_names)

            await self._scrape_players(atp_names, tour="atp")
            await self._scrape_players(wta_names, tour="wta")

            self._last_refresh = time.time()
            self._save_cache()
            log.info("Serve stats refresh complete: %d players", len(self.stats))
        except Exception as e:
            log.error("Serve stats refresh failed: %s", e)

    def _build_name_index(
        self, atp_names: list[str], wta_names: list[str],
    ) -> None:
        """Build word → [(full_ta_name, tour)] index for fuzzy matching.

        Indexes by EVERY word in the name so "Merida" matches
        "Daniel Merida Aguilar" (indexed under "daniel", "merida", "aguilar").
        """
        index: dict[str, list[tuple[str, str]]] = {}
        for names, tour in [(atp_names, "atp"), (wta_names, "wta")]:
            for name in names:
                clean = unidecode(name).lower().strip()
                for word in clean.split():
                    if len(word) < 2:
                        continue
                    if word not in index:
                        index[word] = []
                    index[word].append((name, tour))
        self._name_index = index
        total = len(atp_names) + len(wta_names)
        log.info("Built name index: %d words, %d players", len(index), total)

    async def _fetch_player_list(self, url: str) -> list[str]:
        """Fetch player name list from Tennis Abstract JS file."""
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

        log.info("Scraped serve stats for %d/%d %s players", scraped, len(names), tour)

    async def _fetch_matchmx(
        self, url_name: str, tour: str,
    ) -> dict[str, float] | None:
        """Fetch matchmx data and compute serve stats.

        ATP: inline var matchmx in player-classic.cgi
        WTA: var matchmx in /jsmatches/{Name}.js
        Falls back to the other endpoint if first fails.
        """
        assert self._session is not None

        if tour == "atp":
            urls = [
                ATP_PLAYER_URL.format(name=url_name),
                WTA_MATCHES_URL.format(name=url_name),  # fallback
            ]
        else:
            urls = [
                WTA_MATCHES_URL.format(name=url_name),
                ATP_PLAYER_URL.format(name=url_name),  # fallback
            ]

        for url in urls:
            try:
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        continue
                    text = await resp.text()
            except Exception:
                continue

            match = re.search(
                r'var\s+matchmx\s*=\s*(\[.+?\])\s*;', text, re.DOTALL,
            )
            if not match:
                continue

            try:
                matches = ast.literal_eval(match.group(1))
            except (ValueError, SyntaxError):
                continue

            result = self._compute_serve_stats(matches)
            if result:
                return result

        return None

    @staticmethod
    def _compute_serve_stats(matches: list) -> dict[str, float] | None:
        """Aggregate serve points won by surface from matchmx rows."""
        surface_pts: dict[str, list[int]] = {}

        for row in matches:
            if len(row) <= IDX_SWON + 1:
                continue
            if str(row[0]) < "2024":
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

            if pts < 20:
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
            if total >= 100:
                pct = won / total
                if 0.30 < pct < 0.90:
                    result[surface] = round(pct, 4)

        return result if result else None

    # ------------------------------------------------------------------
    # On-demand lookup for players not in cache
    # ------------------------------------------------------------------

    async def fetch_player_live(self, goalserve_name: str, surface: str) -> float | None:
        """Fetch serve stats for a single player on-demand.

        Used when a player appears in a live match but isn't in our cache.
        Returns the serve % for the given surface, or None if unavailable.
        """
        if not self._session:
            return None

        resolved = self.resolve_ta_name(goalserve_name)
        if resolved:
            ta_name, tour = resolved
            url_name = ta_name.replace(" ", "")
        else:
            # Try raw Goalserve name
            url_name = goalserve_name.replace(" ", "")
            tour = "atp"  # try ATP first

        try:
            stats = await self._fetch_matchmx(url_name, tour)
            if stats:
                key = ta_name.lower() if resolved else goalserve_name.lower()
                self.stats[key] = stats
                if surface in stats:
                    return stats[surface]
        except Exception as e:
            log.debug("Live fetch failed for %s: %s", goalserve_name, e)

        return None
