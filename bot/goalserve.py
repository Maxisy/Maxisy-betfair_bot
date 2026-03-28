"""Goalserve Tennis API poller and score state manager.

Two-tier polling strategy for minimal latency:

1. Discovery poll (every 30s) — hits the full livescore feed to find new
   live matches and their IDs:
     /tennis_scores/home?json=1

2. Fast poll (every 5s) — hits individual match endpoints for matches we
   are actively tracking:
     /tennis_scores/match?id=978179&json=1

This keeps the fast-path response tiny (single match) instead of parsing
the entire day's tournament data on every cycle.

Docs reference: Goalserve Tennis Data Feed, Section 4 (Livescore Feed).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional

import aiohttp

from .config import Config
from .models import ScoreSource, ScoreState

log = logging.getLogger(__name__)

GOALSERVE_BASE = "https://www.goalserve.com/getfeed"
FAST_POLL_INTERVAL = 5.0
DISCOVERY_INTERVAL = 30.0

# Live match statuses per Goalserve docs
LIVE_STATUSES = {"set 1", "set 2", "set 3", "set 4", "set 5"}


def _g(obj: dict, key: str, default: Any = "") -> Any:
    """Get a value from a Goalserve JSON object.

    Goalserve XML-to-JSON conversion may prefix attribute keys with '@'.
    Try both variants.
    """
    return obj.get(f"@{key}", obj.get(key, default))


def _ensure_list(val: Any) -> list:
    """Goalserve returns a dict when there's one item, list when multiple."""
    if val is None:
        return []
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return val
    return []


class GoalservePoller:
    """Polls Goalserve Tennis Livescore Feed for live score data."""

    def __init__(
        self,
        config: Config,
        on_scores_updated: Callable[[], Coroutine] | None = None,
    ) -> None:
        self.config = config
        self._on_scores_updated = on_scores_updated
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # match_id -> ScoreState
        self.scores: dict[str, ScoreState] = {}

        # Match IDs we're actively fast-polling
        self._tracked_ids: set[str] = set()

        # Health tracking
        self.last_success: float = 0.0
        self.consecutive_failures: int = 0
        self.is_degraded: bool = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        log.info("Goalserve poller started (discovery=%.0fs, fast=%.0fs)",
                 DISCOVERY_INTERVAL, FAST_POLL_INTERVAL)

        # Run discovery and fast poll as concurrent tasks
        discovery_task = asyncio.create_task(
            self._discovery_loop(), name="gs_discovery",
        )
        fast_task = asyncio.create_task(
            self._fast_poll_loop(), name="gs_fast",
        )
        await asyncio.gather(discovery_task, fast_task)

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def seconds_since_success(self) -> float:
        if self.last_success == 0:
            return float("inf")
        return time.time() - self.last_success

    # ------------------------------------------------------------------
    # Discovery loop — find new live matches
    # ------------------------------------------------------------------

    async def _discovery_loop(self) -> None:
        """Poll full livescore feed to discover new live matches."""
        while self._running:
            try:
                await self._poll_all()
                self._record_success()
            except Exception as e:
                self._record_failure(e, "discovery")
            await asyncio.sleep(DISCOVERY_INTERVAL)

    async def _poll_all(self) -> None:
        """Fetch full livescore feed and discover/update all live matches."""
        assert self._session is not None
        url = f"{GOALSERVE_BASE}/{self.config.goalserve_api_key}/tennis_scores/home"
        params = {"json": "1"}

        async with self._session.get(
            url, params=params, timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status == 429:
                log.warning("Goalserve rate limited (discovery) — backing off 30s")
                await asyncio.sleep(30)
                return
            if resp.status >= 500:
                raise RuntimeError(f"Goalserve server error: {resp.status}")
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        live_ids = self._parse_full_feed(data)

        # Update tracked set — add new live matches, remove finished ones
        new_ids = live_ids - self._tracked_ids
        gone_ids = self._tracked_ids - live_ids
        if new_ids:
            log.info("Discovered %d new live match(es): %s", len(new_ids), new_ids)
        self._tracked_ids = live_ids

        # Clean up finished matches
        for mid in gone_ids:
            self.scores.pop(mid, None)

        if self._on_scores_updated:
            try:
                await self._on_scores_updated()
            except Exception as e:
                log.error("Score update callback error: %s", e)

    def _parse_full_feed(self, data: dict[str, Any]) -> set[str]:
        """Parse full livescore response. Returns set of live match IDs."""
        scores_root = data.get("scores", data)
        categories = _ensure_list(scores_root.get("category", []))

        live_ids: set[str] = set()

        for category in categories:
            cat_name = _g(category, "name", "")
            surface = self._detect_surface(cat_name)
            best_of = self._detect_best_of(cat_name)

            matches = _ensure_list(category.get("match", []))

            for match in matches:
                match_id = str(_g(match, "id", ""))
                if not match_id:
                    continue

                status = str(_g(match, "status", "")).lower().strip()
                if status not in LIVE_STATUSES:
                    continue

                live_ids.add(match_id)

                # Parse this match into ScoreState
                self._parse_match(match_id, match, cat_name, surface, best_of)

        return live_ids

    # ------------------------------------------------------------------
    # Fast poll loop — update tracked matches individually
    # ------------------------------------------------------------------

    async def _fast_poll_loop(self) -> None:
        """Poll individual match endpoints for tracked matches."""
        while self._running:
            if self._tracked_ids:
                tasks = [
                    self._poll_match(mid) for mid in list(self._tracked_ids)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                if self._on_scores_updated:
                    try:
                        await self._on_scores_updated()
                    except Exception as e:
                        log.error("Score update callback error: %s", e)

            await asyncio.sleep(FAST_POLL_INTERVAL)

    async def _poll_match(self, match_id: str) -> None:
        """Fetch a single match by ID."""
        assert self._session is not None
        url = f"{GOALSERVE_BASE}/{self.config.goalserve_api_key}/tennis_scores/match"
        params = {"id": match_id, "json": "1"}

        try:
            async with self._session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 429:
                    return  # skip this cycle
                if resp.status >= 400:
                    return
                data = await resp.json(content_type=None)

            self._parse_match_response(match_id, data)
            self._record_success()
        except Exception as e:
            self._record_failure(e, f"match {match_id}")

    def _parse_match_response(self, match_id: str, data: dict[str, Any]) -> None:
        """Parse single match response.

        The individual match endpoint may return the match nested under
        scores/category/match or directly — handle both.
        """
        # Try to find the match in the response
        scores_root = data.get("scores", data)
        categories = _ensure_list(scores_root.get("category", []))

        for category in categories:
            cat_name = _g(category, "name", "")
            surface = self._detect_surface(cat_name)
            best_of = self._detect_best_of(cat_name)

            matches = _ensure_list(category.get("match", []))
            for match in matches:
                mid = str(_g(match, "id", ""))
                if mid == match_id:
                    status = str(_g(match, "status", "")).lower().strip()
                    if status in LIVE_STATUSES:
                        self._parse_match(match_id, match, cat_name, surface, best_of)
                    else:
                        # Match is no longer live
                        self._tracked_ids.discard(match_id)
                        self.scores.pop(match_id, None)
                    return

        # If match not found in response, it may have ended
        # Keep it tracked — discovery loop will clean it up

    # ------------------------------------------------------------------
    # Match parsing
    # ------------------------------------------------------------------

    def _parse_match(
        self,
        match_id: str,
        match: dict,
        cat_name: str,
        surface: str,
        best_of: int,
    ) -> None:
        """Parse a single match element into a ScoreState."""
        state = self.scores.get(match_id)
        if state is None:
            state = ScoreState(match_id=match_id)
            self.scores[match_id] = state

        state.tournament = cat_name
        state.surface = surface
        state.best_of = best_of
        state.source = ScoreSource.API
        state.last_updated = time.time()

        # Tiebreak flag
        tb_str = str(_g(match, "tb", "False")).lower()
        state.is_tiebreak = tb_str == "true"

        # Players — two <player> elements per match
        players = _ensure_list(match.get("player", []))
        if len(players) < 2:
            return

        p1 = players[0]
        p2 = players[1]

        state.player1_name = _g(p1, "name", state.player1_name)
        state.player2_name = _g(p2, "name", state.player2_name)

        # Server — each player has serve="True"/"False"
        if str(_g(p1, "serve", "")).lower() == "true":
            state.server = "player1"
        elif str(_g(p2, "serve", "")).lower() == "true":
            state.server = "player2"

        # Set score — use totalscore (sets won by each player)
        p1_sets = self._safe_int(_g(p1, "totalscore", "0"))
        p2_sets = self._safe_int(_g(p2, "totalscore", "0"))
        state.set_score = (p1_sets, p2_sets)

        # Game score in current set — parse s1-s5
        status = str(_g(match, "status", "")).lower().strip()
        self._parse_game_score(p1, p2, status, state)

        # Point score within current game
        self._parse_point_score(p1, p2, state)

    def _parse_game_score(
        self,
        p1: dict,
        p2: dict,
        status: str,
        state: ScoreState,
    ) -> None:
        """Extract game score in the current set from s1-s5 attributes.

        The current set is determined by match status ("set 1" -> s1, etc.).
        Set scores can contain tiebreak scores after "." (e.g. "6.5" = 6 games,
        5 tiebreak pts). We only take the game part (before the dot).
        """
        set_fields = {
            "set 1": "s1", "set 2": "s2", "set 3": "s3",
            "set 4": "s4", "set 5": "s5",
        }
        field_name = set_fields.get(status)
        if not field_name:
            return

        p1_raw = str(_g(p1, field_name, "0"))
        p2_raw = str(_g(p2, field_name, "0"))

        # Strip tiebreak portion: "6.5" -> "6"
        p1_games = self._safe_int(p1_raw.split(".")[0])
        p2_games = self._safe_int(p2_raw.split(".")[0])

        # game_score is stored as (server_games, receiver_games)
        if state.server == "player1":
            state.game_score = (p1_games, p2_games)
        else:
            state.game_score = (p2_games, p1_games)

    def _parse_point_score(
        self,
        p1: dict,
        p2: dict,
        state: ScoreState,
    ) -> None:
        """Extract point score from each player's game_score attribute.

        Regular game: "", "0", "15", "30", "40", "A"
        Tiebreak: actual point numbers "0", "1", "2", etc.
        """
        old_points = state.point_score

        p1_str = str(_g(p1, "game_score", "")).strip()
        p2_str = str(_g(p2, "game_score", "")).strip()

        # If both empty, game is between points or just started
        if not p1_str and not p2_str:
            return

        if state.is_tiebreak:
            p1_pts = self._safe_int(p1_str)
            p2_pts = self._safe_int(p2_str)
        else:
            p1_pts = self._parse_game_point(p1_str)
            p2_pts = self._parse_game_point(p2_str)

        # Store as (server_points, receiver_points)
        if state.server == "player1":
            state.point_score = (p1_pts, p2_pts)
        else:
            state.point_score = (p2_pts, p1_pts)

        # Track points in current game for new-game filter
        if state.point_score == (0, 0) and old_points != (0, 0):
            state.points_in_current_game = 0
        elif state.point_score != old_points:
            state.points_in_current_game += 1

    # ------------------------------------------------------------------
    # Health tracking
    # ------------------------------------------------------------------

    def _record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_success = time.time()
        if self.is_degraded:
            log.info("Goalserve recovered from degraded state")
            self.is_degraded = False

    def _record_failure(self, error: Exception, context: str) -> None:
        self.consecutive_failures += 1
        log.error("Goalserve %s poll failed (#%d): %s",
                  context, self.consecutive_failures, error)
        if self.consecutive_failures >= 3:
            self.is_degraded = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_game_point(s: str) -> int:
        """Convert tennis point string to internal representation (0-4)."""
        s = s.strip().upper()
        mapping = {"": 0, "0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
        return mapping.get(s, 0)

    @staticmethod
    def _safe_int(s: str) -> int:
        s = str(s).strip()
        if not s:
            return 0
        try:
            return int(s)
        except ValueError:
            return 0

    @staticmethod
    def _detect_surface(category_name: str) -> str:
        """Extract surface from category name.

        Format: "Atp - Singles: Sofia (Bulgaria), Hard (Indoor)"
        """
        name_lower = category_name.lower()
        if "clay" in name_lower:
            return "clay"
        if "grass" in name_lower:
            return "grass"
        if "hard" in name_lower:
            return "hard"
        if "carpet" in name_lower:
            return "hard"
        return "hard"

    @staticmethod
    def _detect_best_of(category_name: str) -> int:
        """Grand Slam men's singles is best of 5, everything else best of 3."""
        name_lower = category_name.lower()
        grand_slams = ["australian open", "roland garros", "french open",
                       "wimbledon", "us open"]
        for gs in grand_slams:
            if gs in name_lower:
                if "singles" in name_lower and ("atp" in name_lower or "men" in name_lower):
                    return 5
        return 3
