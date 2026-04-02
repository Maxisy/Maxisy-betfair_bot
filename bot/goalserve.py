"""Goalserve inplay feed poller and score state manager.

Uses the inplay tennis feed (IP-whitelisted, no API key in URL):
  http://inplay.goalserve.com/inplay-tennis.gz

Single HTTP request returns all live tennis matches with scores, stats,
and serve info. Refreshes every second on Goalserve's side. We poll
every 1s to match. Response is ~200ms, scores are ~10% of payload
(rest is bookmaker odds we discard).

Surface data is not in the inplay feed, so we periodically fetch the
score feed to build a tournament→surface lookup:
  https://www.goalserve.com/getfeed/{key}/tennis_scores/home?json=1

State codes (from /dictionaries/states/tennis):
  11113 = Player 1 Serve
  21113 = Player 2 Serve
  11125 = Player 1 Score Point in Tiebreak
  21125 = Player 2 Score Point in Tiebreak
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

INPLAY_URL = "http://inplay.goalserve.com/inplay-tennis.gz"
SCORE_FEED_URL = "https://www.goalserve.com/getfeed/{key}/tennis_scores/home"
POLL_INTERVAL = 1.0
SURFACE_REFRESH_INTERVAL = 300.0  # refresh surface map every 5 minutes

# State codes where a player is serving (normal game)
SERVE_STATES = {11113, 21113}
# State codes for tiebreak activity
TIEBREAK_STATES = {11125, 21125}
# All "live play" states (serve, point scored, tiebreak, etc.)
LIVE_STATES = {
    11113, 21113,  # serve
    11114, 21114,  # score point
    11115, 21115,  # score point (variant)
    11116, 21116,  # double fault
    11117, 21117,  # ace
    11118, 21118,  # break point
    11119, 21119,  # win a game
    11120, 21120,  # statistic
    11121, 21121,  # let 1st serve
    11122, 21122,  # let 2nd serve
    11125, 21125,  # tiebreak point
    11128, 21128,  # point score
}


class GoalservePoller:
    """Polls Goalserve inplay tennis feed for live score data."""

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
        # True after first poll completes — matches seen on first poll are
        # marked as not tracked_from_start since we don't know their history
        self._first_poll_done: bool = False

        # Tournament name -> surface (from score feed)
        self._surface_map: dict[str, str] = {}
        self._surface_last_refresh: float = 0.0

        # Health tracking
        self.last_success: float = 0.0
        self.consecutive_failures: int = 0
        self.is_degraded: bool = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True
        log.info("Goalserve inplay poller started (interval=%.0fs)", POLL_INTERVAL)

        # Initial surface map fetch
        await self._refresh_surface_map()

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
                log.error("Goalserve poll failed (#%d): %s",
                          self.consecutive_failures, e)
                if self.consecutive_failures >= 3:
                    self.is_degraded = True

            # Periodically refresh surface map
            if time.time() - self._surface_last_refresh > SURFACE_REFRESH_INTERVAL:
                await self._refresh_surface_map()

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

        async with self._session.get(
            INPLAY_URL, timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Goalserve inplay error: {resp.status}")
            data = await resp.json(content_type=None)

        # Docs: if root updated_ts hasn't changed in 30s, all data is stale
        feed_ts = data.get("updated_ts")
        if feed_ts:
            try:
                feed_age = time.time() * 1000 - int(feed_ts)
                if feed_age > 30_000:
                    log.warning(
                        "Goalserve feed stale (%.0fs old) — server may be disconnected",
                        feed_age / 1000,
                    )
                    self.is_degraded = True
                    return
            except (ValueError, TypeError):
                pass

        self._parse_feed(data)

        if self._on_scores_updated:
            try:
                await self._on_scores_updated()
            except Exception as e:
                log.error("Score update callback error: %s", e)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_feed(self, data: dict[str, Any]) -> None:
        """Parse inplay feed into ScoreState objects."""
        events = data.get("events", {})
        if not isinstance(events, dict):
            return

        seen_ids: set[str] = set()

        for _key, evt in events.items():
            info = evt.get("info", {})
            match_id = str(info.get("id", ""))
            if not match_id:
                continue

            # Check if match is live
            state_code = self._safe_int(info.get("state"))
            period = str(info.get("period", "")).strip()
            if not period.lower().startswith("set"):
                continue

            seen_ids.add(match_id)

            state = self.scores.get(match_id)
            if state is None:
                state = ScoreState(match_id=match_id)
                self.scores[match_id] = state
                # Only matches that appear AFTER the first poll can be
                # tracked from start — anything on the first poll was
                # already in progress and we have no hold/break history
                state.tracked_from_start = self._first_poll_done

            state.source = ScoreSource.API
            state.last_updated = time.time()

            # Players — from "Name1 vs Name2" in info.name
            match_name = info.get("name", "")
            ti = evt.get("team_info", {})
            state.player1_name = ti.get("home", {}).get("name", state.player1_name)
            state.player2_name = ti.get("away", {}).get("name", state.player2_name)

            # Tournament and surface from info.league
            league = info.get("league", "")
            state.tournament = league
            state.surface = self._lookup_surface(league)
            state.best_of = self._detect_best_of(league)

            # Parse stats
            stats = evt.get("stats", {})
            self._parse_stats(stats, state_code, state)

        # Remove finished/disappeared matches
        for mid in list(self.scores.keys()):
            if mid not in seen_ids:
                del self.scores[mid]

        # After first parse, all subsequent new matches are tracked from start
        if not self._first_poll_done:
            self._first_poll_done = True

    def _parse_stats(
        self, stats: dict[str, Any], state_code: int, state: ScoreState,
    ) -> None:
        """Extract score data from the stats dict."""
        turn = None
        points = None
        total_sets = None
        set_scores: dict[int, dict] = {}

        for _k, v in stats.items():
            if not isinstance(v, dict):
                continue
            name = v.get("name", "")
            if name == "TURN":
                turn = v
            elif name == "POINTS":
                points = v
            elif name == "T":
                total_sets = v
            elif name.startswith("S") and name[1:].isdigit():
                set_scores[int(name[1:])] = v

        # Server — TURN: home=1 means home serves, away=1 means away serves
        if turn is not None:
            if turn.get("home") == 1:
                state.server = "player1"
            elif turn.get("away") == 1:
                state.server = "player2"

        # Set score (sets won)
        if total_sets is not None:
            state.set_score = (
                self._safe_int(total_sets.get("home")),
                self._safe_int(total_sets.get("away")),
            )

        # Game score in current set
        if set_scores:
            current_set_num = max(set_scores.keys())
            cs = set_scores[current_set_num]
            p1_games = self._safe_int(cs.get("home"))
            p2_games = self._safe_int(cs.get("away"))
            # Store as (server_games, receiver_games)
            if state.server == "player1":
                state.game_score = (p1_games, p2_games)
            else:
                state.game_score = (p2_games, p1_games)

        # Tiebreak detection
        state.is_tiebreak = state_code in TIEBREAK_STATES or (
            state.game_score[0] == 6 and state.game_score[1] == 6
        )

        # Point score
        old_points = state.point_score
        if points is not None:
            p1_raw = points.get("home", 0)
            p2_raw = points.get("away", 0)

            if state.is_tiebreak:
                # Tiebreak: points are actual counts (0, 1, 2, ...)
                p1_pts = self._safe_int(p1_raw)
                p2_pts = self._safe_int(p2_raw)
            else:
                # Regular game: points are tennis format (0, 15, 30, 40)
                p1_pts = self._parse_game_point(p1_raw)
                p2_pts = self._parse_game_point(p2_raw)

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

        # Track total points played (for fatigue estimation)
        if state.point_score != old_points:
            state.total_points_played += 1

        # Track break points faced/saved
        # Break point = receiver has 3+ pts (40) and server has < 3 (< 40),
        # or receiver has advantage (Ad). Detect transition through break point.
        self._detect_break_point(state, old_points)

        # Track aces and double faults from state codes
        if state_code == 11117:  # Player 1 ace
            state.player1_aces += 1
        elif state_code == 21117:  # Player 2 ace
            state.player2_aces += 1
        elif state_code == 11116:  # Player 1 double fault
            state.player1_double_faults += 1
        elif state_code == 21116:  # Player 2 double fault
            state.player2_double_faults += 1

        # Detect service game completions (hold or break)
        # Compare raw game scores (p1_games, p2_games) before/after
        if set_scores:
            current_set_num = max(set_scores.keys())
            cs = set_scores[current_set_num]
            p1_games = self._safe_int(cs.get("home"))
            p2_games = self._safe_int(cs.get("away"))
            raw_game = (p1_games, p2_games)

            self._detect_service_game_result(state, raw_game)

    # ------------------------------------------------------------------
    # Surface map from score feed
    # ------------------------------------------------------------------

    async def _refresh_surface_map(self) -> None:
        """Fetch the score feed to build tournament→surface mapping.

        The score feed category names include surface info, e.g.
        "Atp - Singles: Bucharest (Romania), Clay"
        The inplay feed only has "ATP Bucharest" — no surface.
        We build a fuzzy lookup from score feed category names.
        """
        if not self._session:
            return
        try:
            url = SCORE_FEED_URL.format(key=self.config.goalserve_api_key)
            async with self._session.get(
                url, params={"json": "1"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json(content_type=None)

            categories = data.get("scores", {}).get("category", [])
            if isinstance(categories, dict):
                categories = [categories]

            new_map: dict[str, str] = {}
            for cat in categories:
                name = cat.get("@name", cat.get("name", ""))
                if not name:
                    continue
                # Extract surface from end of name: "..., Clay" or "..., Hard"
                surface = "hard"
                name_lower = name.lower()
                if name_lower.endswith("clay"):
                    surface = "clay"
                elif name_lower.endswith("grass"):
                    surface = "grass"
                elif name_lower.endswith("hard"):
                    surface = "hard"
                elif "clay" in name_lower:
                    surface = "clay"
                elif "grass" in name_lower:
                    surface = "grass"

                # Extract the short tournament name for matching
                # "Atp - Singles: Bucharest (Romania), Clay" → "bucharest"
                short = self._extract_city(name)
                if short:
                    new_map[short] = surface

            self._surface_map = new_map
            self._surface_last_refresh = time.time()
            log.info("Surface map refreshed: %d tournaments", len(new_map))

        except Exception as e:
            log.warning("Surface map refresh failed: %s", e)

    @staticmethod
    def _extract_city(category_name: str) -> str:
        """Extract city/location from score feed category name.

        'Atp - Singles: Bucharest (Romania), Clay' → 'bucharest'
        'Itf Men - Singles: M25 Heraklion 2 (Greece), Hard' → 'heraklion'
        """
        # Take part after ':'
        parts = category_name.split(":")
        if len(parts) < 2:
            return ""
        location = parts[-1].strip()
        # Remove surface suffix
        for suffix in [", Clay", ", Hard", ", Grass", ", Carpet"]:
            if location.endswith(suffix):
                location = location[:-len(suffix)]
        # Remove country in parentheses
        if "(" in location:
            location = location[:location.index("(")]
        # Remove tournament tier prefix like "M25 ", "W50 "
        location = location.strip()
        tokens = location.split()
        clean_tokens = []
        for t in tokens:
            # Skip tier codes and trailing numbers
            if t[0].isdigit() or (len(t) <= 4 and t.isalpha() and t.isupper()):
                continue
            clean_tokens.append(t.lower())
        return " ".join(clean_tokens).strip()

    def _lookup_surface(self, league: str) -> str:
        """Look up surface for an inplay league name like 'ATP Bucharest'."""
        league_lower = league.lower()
        # Direct keyword check first
        if "clay" in league_lower:
            return "clay"
        if "grass" in league_lower:
            return "grass"
        if "hard" in league_lower:
            return "hard"
        # Fuzzy match: check both directions —
        # inplay "M25 Santa Margherita" should match map key "santa margherita di pula"
        # and map key "bucharest" should match inplay "ATP Bucharest"
        best_match = ""
        best_surface = "hard"
        for city, surface in self._surface_map.items():
            if not city:
                continue
            if city in league_lower or league_lower in city:
                # Prefer longest match for accuracy
                if len(city) > len(best_match):
                    best_match = city
                    best_surface = surface
        return best_surface

    # ------------------------------------------------------------------
    # In-match tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _record_game_result(state: ScoreState, player: str, held: bool) -> None:
        """Record a service game result for hold/break stats and rolling window."""
        if player == "player1":
            state.player1_service_games += 1
            if held:
                state.player1_service_holds += 1
            state.player1_recent_holds.append(held)
            # Keep only last 5 games for rolling window
            if len(state.player1_recent_holds) > 5:
                state.player1_recent_holds.pop(0)
        elif player == "player2":
            state.player2_service_games += 1
            if held:
                state.player2_service_holds += 1
            state.player2_recent_holds.append(held)
            if len(state.player2_recent_holds) > 5:
                state.player2_recent_holds.pop(0)

    @staticmethod
    def _detect_break_point(state: ScoreState, old_points: tuple[int, int]) -> None:
        """Detect break point situations and track saves/conversions.

        Break point = receiver at match point in the game (3+ pts, leading server).
        In tennis notation: receiver at 40 when server at 0/15/30, or Ad receiver.
        Point encoding: 0=0, 1=15, 2=30, 3=40, 4=Ad.
        """
        if state.point_score == old_points:
            return  # no point change

        # Check if the OLD point score was a break point
        # (server_pts, receiver_pts) — break point when receiver >= 3 and receiver > server
        # or when receiver == 3 and server <= 2 (30-40, 15-40, 0-40)
        old_server, old_receiver = old_points
        was_break_point = (
            (old_receiver >= 3 and old_server <= 2) or  # 0-40, 15-40, 30-40
            (old_receiver == 4 and old_server == 3)      # Ad receiver
        )
        if not was_break_point:
            return

        # Determine who was serving at the old point
        server = state._prev_server or state.server

        # The point was a break point — did the server save it?
        new_server, new_receiver = state.point_score
        # Server saved if: server gained a point or we're back to deuce
        server_saved = (new_server > old_server) or (new_server == 3 and new_receiver == 3)

        if server == "player1":
            state.player1_bp_faced += 1
            if server_saved:
                state.player1_bp_saved += 1
        elif server == "player2":
            state.player2_bp_faced += 1
            if server_saved:
                state.player2_bp_saved += 1

    @staticmethod
    def _detect_service_game_result(state: ScoreState, raw_game: tuple[int, int]) -> None:
        """Detect when a service game completes and record hold/break.

        Compares current raw game score (p1_games, p2_games) against
        previous to detect game transitions. Uses _prev_server to know
        who was serving the completed game.
        """
        prev_raw = state._prev_game_score
        prev_server = state._prev_server

        # Check for set change (games reset)
        set_changed = state.set_score != state._prev_set_score

        if set_changed and state._prev_set_score != (0, 0):
            p1_prev_sets, p2_prev_sets = state._prev_set_score
            p1_sets, p2_sets = state.set_score
            if p1_sets > p1_prev_sets:
                set_winner = "player1"
            elif p2_sets > p2_prev_sets:
                set_winner = "player2"
            else:
                set_winner = ""

            if prev_server and set_winner:
                held = (prev_server == set_winner)
                GoalservePoller._record_game_result(state, prev_server, held)

        elif not set_changed and prev_raw != (0, 0):
            prev_total = prev_raw[0] + prev_raw[1]
            curr_total = raw_game[0] + raw_game[1]

            if curr_total == prev_total + 1 and prev_server:
                p1_gained = raw_game[0] > prev_raw[0]
                p2_gained = raw_game[1] > prev_raw[1]

                if prev_server == "player1":
                    held = p1_gained
                elif prev_server == "player2":
                    held = p2_gained
                else:
                    held = False
                GoalservePoller._record_game_result(state, prev_server, held)

        # Update previous state for next comparison
        state._prev_game_score = raw_game
        state._prev_set_score = state.set_score
        state._prev_server = state.server

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_game_point(val: Any) -> int:
        """Convert tennis point (0/15/30/40/A) to internal representation (0-4)."""
        s = str(val).strip().upper()
        mapping = {"0": 0, "15": 1, "30": 2, "40": 3, "A": 4, "AD": 4}
        return mapping.get(s, 0)

    @staticmethod
    def _safe_int(val: Any) -> int:
        if val is None:
            return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _detect_best_of(league: str) -> int:
        """Grand Slam men's singles is best of 5, everything else best of 3."""
        name_lower = league.lower()
        grand_slams = ["australian open", "roland garros", "french open",
                       "wimbledon", "us open"]
        for gs in grand_slams:
            if gs in name_lower:
                return 5
        return 3
