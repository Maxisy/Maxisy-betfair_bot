"""Tennis Abstract serve stats loader — on-demand model.

Fetches serve win percentages per player per surface from tennisabstract.com.

Data sources:
  - Name index: /jsplayers/curr_rank_atp.js and curr_rank_wta.js
  - Match data (ATP): /cgi-bin/player-classic.cgi?p={FirstLast} (inline matchmx)
  - Match data (WTA): /jsmatches/{FirstLast}.js (external matchmx)
    Fields at indices: [2]=surface, [23]=pts, [25]=fwon, [26]=swon
    Service points won % = (fwon + swon) / pts

Player names in Goalserve are often truncated (e.g. "Daniel Merida" vs
"Daniel Merida Aguilar" on TA). We build a name index keyed by every word
in the name and fuzzy-match Goalserve names against it.

On startup: load disk cache + fetch name index (~2 fast requests).
Per player: fetch on-demand when first seen in a live match, cache forever.
No bulk scrape — avoids rate limiting and wasted requests.
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
NAME_INDEX_REFRESH = 24 * 3600  # refresh name index daily

TA_BASE = "https://www.tennisabstract.com"
ATP_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_atp.js"
WTA_RANKS_URL = f"{TA_BASE}/jsplayers/curr_rank_wta.js"
ATP_ELO_URL = f"{TA_BASE}/reports/atp_elo_ratings.html"
WTA_ELO_URL = f"{TA_BASE}/reports/wta_elo_ratings.html"
ATP_PLAYER_URL = f"{TA_BASE}/cgi-bin/player-classic.cgi?p={{name}}"
WTA_MATCHES_URL = f"{TA_BASE}/jsmatches/{{name}}.js"

# matchmx field indices — serve block
IDX_SURFACE = 2
IDX_PTS = 23
IDX_FWON = 25
IDX_SWON = 26
# matchmx field indices — return block (opponent's serve stats)
IDX_RET_PTS = 32       # opponent total service points
IDX_OPP_FWON = 34      # opponent 1st serve points won
IDX_OPP_SWON = 35      # opponent 2nd serve points won
# return points won = IDX_RET_PTS - IDX_OPP_FWON - IDX_OPP_SWON


class ServeStatsLoader:
    """Load and cache serve win percentages per player per surface.

    Fetch model: on-demand per player, not bulk.
    - Name index (ATP+WTA rank lists) refreshed daily (~2 HTTP requests)
    - Individual player stats fetched on first encounter, cached to disk
    - Background fetch so trading loop is never blocked
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        # "firstname lastname" (lower) -> {"hard": 0.63, "clay": 0.60, ...}
        self.stats: dict[str, dict[str, float]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        # Name index: name_word (lower) -> [(full_ta_name, tour)]
        self._name_index: dict[str, list[tuple[str, str]]] = {}
        self._name_index_updated: float = 0.0
        # Track in-flight fetches to avoid duplicate requests
        self._pending: dict[str, asyncio.Task] = {}
        # Players not found on TA — retry after TTL (may have been rate-limited)
        self._not_found: dict[str, float] = {}  # name -> timestamp
        self._not_found_ttl: float = 3600.0  # retry after 1 hour
        # Elo ratings: name (lower) -> {"elo": 1800, "hard": 1750, "clay": 1700, "grass": 1650}
        self.elo_ratings: dict[str, dict[str, float]] = {}
        # Dirty flag for cache saves
        self._dirty = False

    async def start(self) -> None:
        """Load cache, fetch name index, then run cache-save loop."""
        self._load_cached()
        self._session = aiohttp.ClientSession()
        self._running = True

        # Fetch name index immediately (2 fast requests)
        await self._refresh_name_index()

        # Periodic loop: save cache + refresh name index
        while self._running:
            await asyncio.sleep(300)  # save cache every 5 min if dirty
            if self._dirty:
                self._save_cache()
                self._dirty = False
            if time.time() - self._name_index_updated > NAME_INDEX_REFRESH:
                await self._refresh_name_index()

    async def stop(self) -> None:
        self._running = False
        # Cancel pending fetches
        for task in self._pending.values():
            task.cancel()
        self._pending.clear()
        # Final cache save
        if self._dirty:
            self._save_cache()
        if self._session:
            await self._session.close()
            self._session = None

    def _find_player_stats(self, player_name: str) -> dict[str, float] | None:
        """Find the stats dict for a player by name matching.

        Returns the stats dict or None if player not in cache at all.
        """
        name_clean = unidecode(player_name).lower().strip()

        # 1. Exact full name
        if name_clean in self.stats:
            return self.stats[name_clean]

        # 2. Goalserve name is prefix of TA name
        for key, val in self.stats.items():
            if key.startswith(name_clean):
                return val

        # 3. Last-word surname + first name
        parts = name_clean.split()
        if parts:
            surname = parts[-1]
            first = parts[0]
            for key, val in self.stats.items():
                kparts = key.split()
                if kparts and kparts[-1] == surname and kparts[0] == first:
                    return val
            for key, val in self.stats.items():
                kparts = key.split()
                if kparts and kparts[-1] == surname:
                    return val

        return None

    def _get_stat(self, player_name: str, stat_key: str) -> float | None:
        """Get a specific stat for the exact surface requested.

        Returns None if player not found or has no data for this surface.
        No cross-surface fallback — wrong surface data creates fake edges.
        """
        stats = self._find_player_stats(player_name)
        if stats is None:
            return None

        if stat_key in stats:
            return stats[stat_key]

        return None

    def get_serve_pct(self, player_name: str, surface: str) -> float | None:
        """Get serve win % for a player on a surface.

        Falls back to other surfaces if exact surface not available.
        Returns None if player not found at all.
        """
        return self._get_stat(player_name, surface)

    def get_return_pct(self, player_name: str, surface: str) -> float | None:
        """Get return win % for a player on a surface.

        Falls back to other surfaces if exact surface not available.
        Returns None if not available.
        """
        return self._get_stat(player_name, f"{surface}_ret")

    def get_elo(self, player_name: str, surface: str) -> float | None:
        """Get surface-specific Elo for a player.

        Falls back to overall Elo if surface-specific not available.
        Returns None if player not in Elo database.
        """
        name_clean = unidecode(player_name).lower().strip()

        # Try exact name
        elo = self.elo_ratings.get(name_clean)
        if not elo:
            # Try prefix match
            for key, val in self.elo_ratings.items():
                if key.startswith(name_clean):
                    elo = val
                    break
        if not elo:
            # Try surname + first name
            parts = name_clean.split()
            if parts:
                surname = parts[-1]
                first = parts[0]
                for key, val in self.elo_ratings.items():
                    kparts = key.split()
                    if kparts and kparts[-1] == surname and kparts[0] == first:
                        elo = val
                        break

        if not elo:
            return None

        # Surface-specific Elo, fallback to overall
        surface_key = {"hard": "hard", "clay": "clay", "grass": "grass"}.get(surface)
        if surface_key and surface_key in elo:
            return elo[surface_key]
        return elo.get("elo")

    def ensure_player(self, goalserve_name: str, surface: str) -> None:
        """Trigger background fetch for a player if not cached.

        Non-blocking — data will be available on next Goalserve poll cycle.
        """
        if not self._session or not self._running:
            return

        name_clean = unidecode(goalserve_name).lower().strip()

        # Already have data
        if self.get_serve_pct(goalserve_name, surface) is not None:
            return

        # Recently confirmed not on TA — retry after TTL
        not_found_ts = self._not_found.get(name_clean)
        if not_found_ts and (time.time() - not_found_ts) < self._not_found_ttl:
            return

        # Already fetching
        if name_clean in self._pending:
            return

        # Launch background fetch
        task = asyncio.create_task(
            self._fetch_and_cache(goalserve_name, surface),
        )
        self._pending[name_clean] = task
        task.add_done_callback(
            lambda t, k=name_clean: self._pending.pop(k, None),
        )

    async def _fetch_and_cache(self, goalserve_name: str, surface: str) -> None:
        """Fetch a single player's stats and cache the result."""
        name_clean = unidecode(goalserve_name).lower().strip()

        resolved = self.resolve_ta_name(goalserve_name)
        if resolved:
            ta_name, tour = resolved
            url_names = [ta_name.replace(" ", "")]
        else:
            raw = goalserve_name.replace(" ", "")
            title = goalserve_name.title().replace(" ", "")
            url_names = [raw] if raw == title else [raw, title]
            tour = "atp"

        for url_name in url_names:
            try:
                stats = await self._fetch_matchmx(url_name, tour)
                if stats:
                    key = ta_name.lower() if resolved else name_clean
                    self.stats[key] = stats
                    self._dirty = True
                    log.info(
                        "Fetched serve stats for %s (%s): %s",
                        goalserve_name, url_name,
                        {s: f"{p:.1%}" for s, p in stats.items()},
                    )
                    return
            except Exception as e:
                log.debug("Fetch failed for %s (%s): %s", goalserve_name, url_name, e)

        # Not found on TA — will retry after TTL
        self._not_found[name_clean] = time.time()
        log.info("Player not found on TA: %s (will retry in %.0fm)", goalserve_name, self._not_found_ttl / 60)

    def resolve_ta_name(self, goalserve_name: str) -> tuple[str, str] | None:
        """Resolve a Goalserve player name to (full_ta_name, tour).

        Returns original TA casing (e.g. "Daniel Altmaier") for URL construction.
        """
        clean = unidecode(goalserve_name).lower().strip()
        words = clean.split()
        if not words:
            return None

        candidates: dict[str, str] = {}
        for word in words:
            for ta_name, tour in self._name_index.get(word, []):
                candidates[ta_name.lower()] = tour

        if not candidates:
            return None

        best_lower = None
        best_score = -1
        for ta_lower, tour in candidates.items():
            if ta_lower == clean:
                best_lower = ta_lower
                best_score = 1000
                break
            if ta_lower.startswith(clean):
                score = 100 + len(clean)
            else:
                score = sum(1 for w in words if w in ta_lower.split())
            if score > best_score:
                best_score = score
                best_lower = ta_lower

        if best_lower is None or best_score < 2:
            return None

        # Find original casing from the name index
        for word in words:
            for ta_name, tour in self._name_index.get(word, []):
                if ta_name.lower() == best_lower:
                    return (ta_name, tour)

        # Fallback (shouldn't happen — index always has original casing)
        return (best_lower, candidates.get(best_lower, "atp"))

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _load_cached(self) -> None:
        if STATS_FILE.exists():
            try:
                data = json.loads(STATS_FILE.read_text())
                self.stats = data.get("stats", {})
                # Migrate from old list format to dict with timestamps
                not_found_raw = data.get("not_found", [])
                if isinstance(not_found_raw, list):
                    self._not_found = {name: 0.0 for name in not_found_raw}
                else:
                    self._not_found = not_found_raw
                self.elo_ratings = data.get("elo_ratings", {})
                stored_index = data.get("name_index", {})
                if stored_index:
                    self._name_index = {
                        k: [(n, t) for n, t in v]
                        for k, v in stored_index.items()
                    }
                    self._name_index_updated = data.get("name_index_updated", 0)
                log.info(
                    "Loaded cache: %d players, %d not-found, %d index words, %d elo",
                    len(self.stats), len(self._not_found),
                    len(self._name_index), len(self.elo_ratings),
                )
            except Exception as e:
                log.warning("Failed to load cached stats: %s", e)

    def _save_cache(self) -> None:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats,
            "not_found": self._not_found,
            "elo_ratings": self.elo_ratings,
            "name_index": {
                k: [[n, t] for n, t in v]
                for k, v in self._name_index.items()
            },
            "name_index_updated": self._name_index_updated,
        }
        STATS_FILE.write_text(json.dumps(data, indent=2))
        log.info("Saved cache: %d players", len(self.stats))

    # ------------------------------------------------------------------
    # Name index (lightweight — just rank lists, no player scraping)
    # ------------------------------------------------------------------

    async def _refresh_name_index(self) -> None:
        """Fetch ATP + WTA rank lists and Elo ratings to build name index."""
        assert self._session is not None
        try:
            atp_names = await self._fetch_player_list(ATP_RANKS_URL)
            wta_names = await self._fetch_player_list(WTA_RANKS_URL)
            self._build_name_index(atp_names, wta_names)
            self._name_index_updated = time.time()
            self._dirty = True
            log.info(
                "Name index refreshed: %d ATP + %d WTA players",
                len(atp_names), len(wta_names),
            )
        except Exception as e:
            log.warning("Name index refresh failed: %s", e)

        # Fetch Elo ratings
        try:
            atp_elo = await self._fetch_elo_ratings(ATP_ELO_URL)
            wta_elo = await self._fetch_elo_ratings(WTA_ELO_URL)
            self.elo_ratings.update(atp_elo)
            self.elo_ratings.update(wta_elo)
            self._dirty = True
            log.info("Elo ratings loaded: %d ATP + %d WTA", len(atp_elo), len(wta_elo))
        except Exception as e:
            log.warning("Elo ratings fetch failed: %s", e)

    def _build_name_index(
        self, atp_names: list[str], wta_names: list[str],
    ) -> None:
        """Build word -> [(full_ta_name, tour)] index for fuzzy matching."""
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

    async def _fetch_elo_ratings(self, url: str) -> dict[str, dict[str, float]]:
        """Scrape Elo ratings from Tennis Abstract HTML table.

        Returns dict: name (lower) -> {"elo": 1800.0, "hard": 1750.0, "clay": 1700.0, "grass": 1650.0}
        """
        assert self._session is not None
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                if resp.status != 200:
                    log.warning("Elo fetch failed: %d from %s", resp.status, url)
                    return {}
                text = await resp.text()
        except Exception as e:
            log.warning("Elo fetch error: %s", e)
            return {}

        ratings: dict[str, dict[str, float]] = {}
        # Parse HTML table rows
        # Columns: Elo Rank, Player, Age, Elo, _, hElo Rank, hElo, cElo Rank, cElo, gElo Rank, gElo, ...
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', text, re.DOTALL)
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 11:
                continue

            # Extract player name (strip HTML tags and &nbsp;)
            name_raw = re.sub(r'<[^>]+>', '', cells[1]).replace('&nbsp;', ' ').strip()
            if not name_raw or not name_raw[0].isalpha():
                continue

            name_clean = unidecode(name_raw).lower().strip()

            try:
                elo = float(cells[3].strip()) if cells[3].strip() else 0.0
                h_elo = float(cells[6].strip()) if cells[6].strip() else 0.0
                c_elo = float(cells[8].strip()) if cells[8].strip() else 0.0
                g_elo = float(cells[10].strip()) if cells[10].strip() else 0.0
            except (ValueError, IndexError):
                continue

            if elo < 1000:
                continue  # invalid

            entry: dict[str, float] = {"elo": elo}
            if h_elo > 1000:
                entry["hard"] = h_elo
            if c_elo > 1000:
                entry["clay"] = c_elo
            if g_elo > 1000:
                entry["grass"] = g_elo
            ratings[name_clean] = entry

        return ratings

    # ------------------------------------------------------------------
    # Individual player fetch
    # ------------------------------------------------------------------

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
                WTA_MATCHES_URL.format(name=url_name),
            ]
        else:
            urls = [
                WTA_MATCHES_URL.format(name=url_name),
                ATP_PLAYER_URL.format(name=url_name),
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
        """Aggregate serve and return points won by surface from matchmx rows.

        Returns dict like {"hard": 0.63, "hard_ret": 0.32, "clay": 0.60, ...}
        Keys without '_ret' suffix are serve win %, with '_ret' are return win %.
        """
        # [serve_won, serve_total, return_won, return_total]
        surface_pts: dict[str, list[int]] = {}

        for row in matches:
            if len(row) <= IDX_OPP_SWON + 1:
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
                ret_pts = int(row[IDX_RET_PTS]) if row[IDX_RET_PTS] else 0
                opp_fwon = int(row[IDX_OPP_FWON]) if row[IDX_OPP_FWON] else 0
                opp_swon = int(row[IDX_OPP_SWON]) if row[IDX_OPP_SWON] else 0
            except (ValueError, TypeError):
                continue

            if pts < 20:
                continue

            serve_won = fwon + swon
            ret_won = ret_pts - opp_fwon - opp_swon

            if surface not in surface_pts:
                surface_pts[surface] = [0, 0, 0, 0]
            surface_pts[surface][0] += serve_won
            surface_pts[surface][1] += pts
            if ret_pts >= 20 and ret_won >= 0:
                surface_pts[surface][2] += ret_won
                surface_pts[surface][3] += ret_pts

        if not surface_pts:
            return None

        result: dict[str, float] = {}
        for surface, (s_won, s_total, r_won, r_total) in surface_pts.items():
            if s_total >= 100:
                pct = s_won / s_total
                if 0.30 < pct < 0.90:
                    result[surface] = round(pct, 4)
            if r_total >= 100:
                ret_pct = r_won / r_total
                if 0.10 < ret_pct < 0.60:
                    result[f"{surface}_ret"] = round(ret_pct, 4)

        return result if result else None
