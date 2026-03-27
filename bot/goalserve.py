"""Goalserve Tennis API poller and score state manager.

Polls every 5 seconds for live match data. Updates ScoreState objects.
Falls back to odds-inference mode when Goalserve is unavailable.
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
POLL_INTERVAL = 5.0


class GoalservePoller:
    """Polls Goalserve Tennis API for live score data."""

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

        # Health tracking
        self.last_success: float = 0.0
        self.consecutive_failures: int = 0
        self.is_degraded: bool = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        log.info("Goalserve poller started")
        while self._running:
            try:
                await self._poll()
                self.consecutive_failures = 0
                self.last_success = time.time()
                if self.is_degraded:
                    log.info("Goalserve recovered from degraded state")
                    self.is_degraded = False
            except Exception as e:
                self.consecutive_failures += 1
                log.error("Goalserve poll failed (#%d): %s", self.consecutive_failures, e)
                if self.consecutive_failures >= 3:
                    self.is_degraded = True

            await asyncio.sleep(POLL_INTERVAL)

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

    async def _poll(self) -> None:
        assert self._session is not None
        url = f"{GOALSERVE_BASE}/{self.config.goalserve_api_key}/tennis/livescore"
        params = {"json": "1"}

        async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 429:
                log.warning("Goalserve rate limited — backing off 30s")
                await asyncio.sleep(30)
                return
            if resp.status >= 500:
                raise RuntimeError(f"Goalserve server error: {resp.status}")
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        self._parse_scores(data)

        if self._on_scores_updated:
            try:
                await self._on_scores_updated()
            except Exception as e:
                log.error("Score update callback error: %s", e)

    def _parse_scores(self, data: dict[str, Any]) -> None:
        """Parse Goalserve JSON response into ScoreState objects."""
        tournaments = data.get("scores", {}).get("category", [])
        if isinstance(tournaments, dict):
            tournaments = [tournaments]

        seen_ids: set[str] = set()

        for tournament in tournaments:
            tournament_name = tournament.get("name", "")
            surface = self._detect_surface(tournament_name, tournament)

            matches = tournament.get("match", [])
            if isinstance(matches, dict):
                matches = [matches]

            for match in matches:
                match_id = str(match.get("id", ""))
                if not match_id:
                    continue
                seen_ids.add(match_id)

                # Only process live matches
                status = match.get("status", "").lower()
                if status not in ("inprogress", "in progress", "live", "started"):
                    continue

                state = self.scores.get(match_id)
                if state is None:
                    state = ScoreState(match_id=match_id)
                    self.scores[match_id] = state

                state.tournament = tournament_name
                state.surface = surface
                state.source = ScoreSource.API
                state.last_updated = time.time()

                # Players
                p1 = match.get("localteam", {})
                p2 = match.get("visitorteam", {})
                state.player1_name = p1.get("name", state.player1_name)
                state.player2_name = p2.get("name", state.player2_name)

                # Match format
                best_of = match.get("bestof", "3")
                state.best_of = int(best_of) if str(best_of).isdigit() else 3

                # Server
                serving = match.get("serving", "")
                if serving:
                    if serving == "1" or serving.lower() == state.player1_name.lower():
                        state.server = "player1"
                    else:
                        state.server = "player2"

                # Set score
                self._parse_set_score(match, state)

                # Game score within current set
                self._parse_game_score(match, state)

                # Point score
                self._parse_point_score(match, state)

        # Remove finished matches
        for mid in list(self.scores.keys()):
            if mid not in seen_ids:
                del self.scores[mid]

    def _parse_set_score(self, match: dict, state: ScoreState) -> None:
        """Extract sets won by each player."""
        sets_p1 = 0
        sets_p2 = 0
        sets_data = match.get("sets", {}).get("set", [])
        if isinstance(sets_data, dict):
            sets_data = [sets_data]
        for s in sets_data:
            s1 = int(s.get("score1", 0) or 0)
            s2 = int(s.get("score2", 0) or 0)
            if s1 > s2:
                sets_p1 += 1
            elif s2 > s1:
                sets_p2 += 1
            # else: current set in progress, don't count
        # The last set in the list is the current set
        state.set_score = (sets_p1, sets_p2)

    def _parse_game_score(self, match: dict, state: ScoreState) -> None:
        """Extract game score in current set."""
        sets_data = match.get("sets", {}).get("set", [])
        if isinstance(sets_data, dict):
            sets_data = [sets_data]
        if sets_data:
            current = sets_data[-1]
            g1 = int(current.get("score1", 0) or 0)
            g2 = int(current.get("score2", 0) or 0)
            # Convert to server/receiver perspective
            if state.server == "player1":
                state.game_score = (g1, g2)
            else:
                state.game_score = (g2, g1)

    def _parse_point_score(self, match: dict, state: ScoreState) -> None:
        """Extract point score within current game."""
        point_str = match.get("game", {}).get("score", "")
        if not point_str:
            point_str = match.get("pointscore", "")

        old_points = state.point_score
        if point_str:
            parts = point_str.replace(" ", "").split("-")
            if len(parts) == 2:
                p1_pts = self._parse_point(parts[0])
                p2_pts = self._parse_point(parts[1])
                if state.server == "player1":
                    state.point_score = (p1_pts, p2_pts)
                else:
                    state.point_score = (p2_pts, p1_pts)

        # Track points in current game for new-game filter
        if state.point_score == (0, 0) and old_points != (0, 0):
            state.points_in_current_game = 0
        elif state.point_score != old_points:
            state.points_in_current_game += 1

    @staticmethod
    def _parse_point(s: str) -> int:
        """Convert point string to internal representation (0-4)."""
        s = s.strip().upper()
        mapping = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
        return mapping.get(s, 0)

    @staticmethod
    def _detect_surface(tournament_name: str, tournament: dict) -> str:
        name_lower = tournament_name.lower()
        surface = tournament.get("surface", "").lower()
        if surface in ("hard", "clay", "grass"):
            return surface
        if "clay" in name_lower or "roland" in name_lower:
            return "clay"
        if "grass" in name_lower or "wimbledon" in name_lower:
            return "grass"
        return "hard"
