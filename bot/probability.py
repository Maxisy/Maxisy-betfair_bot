"""Four-layer Markov chain probability model for tennis.

Layer 1: Point  (input: p = serve win %)
Layer 2: Game   (from point score → game win probability)
Layer 3: Set    (from game score → set win probability, with tiebreak)
Layer 4: Match  (from set score → match win probability)
"""

from __future__ import annotations

from functools import lru_cache

from .models import ScoreState
from .ticks import nearest_tick


# ---------------------------------------------------------------------------
# Layer 2 — Game win probability from current point score
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def prob_win_game(p: float, server_pts: int, receiver_pts: int) -> float:
    """Probability the server wins the current game from point score (s, r).

    Points encoded: 0=0, 1=15, 2=30, 3=40, 4=Ad.
    """
    # Terminal states
    if server_pts >= 4 and server_pts - receiver_pts >= 2:
        return 1.0  # server already won
    if receiver_pts >= 4 and receiver_pts - server_pts >= 2:
        return 0.0  # receiver already won
    if server_pts == 4 and receiver_pts == 4:
        # Back to deuce equivalent
        return _prob_win_from_deuce(p)

    # Deuce (3-3)
    if server_pts >= 3 and receiver_pts >= 3:
        if server_pts == receiver_pts:
            return _prob_win_from_deuce(p)
        elif server_pts > receiver_pts:
            # Ad server (4-3)
            return p * 1.0 + (1 - p) * _prob_win_from_deuce(p)
        else:
            # Ad receiver (3-4)
            return p * _prob_win_from_deuce(p) + (1 - p) * 0.0

    # Normal points: recurse
    return (
        p * prob_win_game(p, server_pts + 1, receiver_pts)
        + (1 - p) * prob_win_game(p, server_pts, receiver_pts + 1)
    )


def _prob_win_from_deuce(p: float) -> float:
    """Closed-form probability server wins from deuce."""
    return (p * p) / (p * p + (1 - p) * (1 - p))


# ---------------------------------------------------------------------------
# Layer 2b — Tiebreak win probability
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8192)
def prob_win_tiebreak(p_server: float, p_returner: float,
                       s: int, r: int) -> float:
    """Probability that the player who served first in the tiebreak wins it.

    p_server: prob the tiebreak-first-server wins a point on their serve.
    p_returner: prob the other player wins a point on their serve.
    Service pattern: server serves point 1, then alternate every 2 points.

    Uses closed-form for deuce-equivalent states (s >= 6 and r >= 6 with s == r)
    to avoid infinite recursion.
    """
    # Terminal
    if s >= 7 and s - r >= 2:
        return 1.0
    if r >= 7 and r - s >= 2:
        return 0.0

    # At tiebreak-deuce (6-6, 7-7, etc.) use closed form to prevent
    # infinite recursion. From any even deuce state, we need to win
    # 2 consecutive mini-points (one serve each side in a cycle of 2).
    if s >= 6 and r >= 6 and s == r:
        # Two points will be played: one by each server.
        # Determine who serves first of this pair.
        total = s + r
        first_serves = ((total - 1) // 2) % 2 == 1  # True = p_server serves
        # pa = prob first player wins point when first-of-pair serves
        # pb = prob first player wins point when second-of-pair serves
        if first_serves:
            pa = p_server           # first player is serving
            pb = 1.0 - p_returner   # second player is serving, first player returning
        else:
            pa = 1.0 - p_returner   # second player is serving
            pb = p_server            # first player is serving
        # Prob of winning both points: pa * pb
        # Prob of losing both: (1-pa)*(1-pb)
        # Otherwise back to deuce
        p_win_both = pa * pb
        p_lose_both = (1 - pa) * (1 - pb)
        denom = p_win_both + p_lose_both
        if denom < 1e-15:
            return 0.5
        return p_win_both / denom

    # Determine who serves this point
    # Tiebreak pattern: A serves point 0, then alternate every 2 points
    # A: 0, 3,4, 7,8, 11,12 ...  B: 1,2, 5,6, 9,10 ...
    total_points = s + r
    if total_points == 0:
        first_player_serving = True
    else:
        first_player_serving = ((total_points - 1) // 2) % 2 == 1

    # p = probability the FIRST player (s) wins this point
    # When first player serves: they win with prob p_server
    # When second player serves: first player wins with prob (1 - p_returner)
    p = p_server if first_player_serving else (1.0 - p_returner)

    return (
        p * prob_win_tiebreak(p_server, p_returner, s + 1, r)
        + (1 - p) * prob_win_tiebreak(p_server, p_returner, s, r + 1)
    )


# ---------------------------------------------------------------------------
# Layer 3 — Set win probability from current game score
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def prob_win_set(p_server: float, p_returner: float,
                  server_games: int, receiver_games: int) -> float:
    """Probability the current server's player wins the set from game score (s, r).

    Assumes server changes every game. At the start of each game, the current
    server's game-win probability is computed from their serve %.

    p_server: probability current server wins a point on their serve.
    p_returner: probability the OTHER player wins a point on THEIR serve
                (i.e., when they are serving).
    """
    # Terminal states
    if server_games >= 6 and server_games - receiver_games >= 2:
        return 1.0
    if receiver_games >= 6 and receiver_games - server_games >= 2:
        return 0.0

    # Tiebreak at 6-6
    if server_games == 6 and receiver_games == 6:
        return prob_win_tiebreak(p_server, p_returner, 0, 0)

    # Current game is a service game for 'server'
    g = prob_win_game(p_server, 0, 0)  # prob server holds

    # If server wins this game: score becomes (s+1, r), and service switches.
    # When service switches, the new server is the 'receiver' with p_returner.
    p_after_hold = prob_win_set(p_returner, p_server,
                                 receiver_games, server_games + 1)
    # Note: we swap perspective because the other player now serves.
    # The return value is from the NEW server's perspective, so we need 1 - that
    # to get the original server's perspective.
    p_after_hold = 1.0 - p_after_hold

    # If server loses this game (broken): service switches
    p_after_break = prob_win_set(p_returner, p_server,
                                  receiver_games + 1, server_games)
    p_after_break = 1.0 - p_after_break

    return g * p_after_hold + (1 - g) * p_after_break


# ---------------------------------------------------------------------------
# Layer 4 — Match win probability from current set score
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def prob_win_match(p_server: float, p_returner: float,
                    server_sets: int, receiver_sets: int,
                    sets_to_win: int) -> float:
    """Probability the current server's player wins the match.

    sets_to_win: 2 for best-of-3, 3 for best-of-5.
    """
    if server_sets >= sets_to_win:
        return 1.0
    if receiver_sets >= sets_to_win:
        return 0.0

    # Probability server wins this set (starting from 0-0 games in new set)
    s = prob_win_set(p_server, p_returner, 0, 0)

    # If server wins the set, they still serve first next set
    # (simplification — in reality it depends on who served last in the set)
    p_win_set = s * prob_win_match(p_server, p_returner,
                                     server_sets + 1, receiver_sets, sets_to_win)
    p_lose_set = (1 - s) * prob_win_match(p_server, p_returner,
                                            server_sets, receiver_sets + 1, sets_to_win)
    return p_win_set + p_lose_set


# ---------------------------------------------------------------------------
# Full model: from ScoreState → match win probability + model odds
# ---------------------------------------------------------------------------

def calculate_model_odds(state: ScoreState) -> tuple[float, float]:
    """Calculate model probability and decimal odds for the SERVER winning.

    Returns (probability, model_odds_decimal).
    model_odds is rounded to nearest Betfair tick.
    """
    p_serve = state.server_serve_pct
    p_return_serve = state.receiver_serve_pct  # other player's serve %

    # Layer 2: probability server wins current game
    g = prob_win_game(p_serve, state.point_score[0], state.point_score[1])

    # Layer 3: probability server wins current set from current game score
    # We need to handle the current game being in progress.
    # First get prob of winning set assuming current game is won or lost,
    # weighted by prob of winning current game.
    sg, rg = state.game_score

    # If server wins this game → (sg+1, rg), service switches
    if sg + 1 >= 6 and (sg + 1) - rg >= 2:
        p_set_after_hold = 1.0
    elif sg + 1 == 6 and rg == 6:
        p_set_after_hold = prob_win_tiebreak(p_serve, p_return_serve, 0, 0)
    else:
        p_set_after_hold = 1.0 - prob_win_set(p_return_serve, p_serve, rg, sg + 1)

    # If server loses this game → (sg, rg+1), service switches
    if rg + 1 >= 6 and (rg + 1) - sg >= 2:
        p_set_after_break = 0.0
    elif rg + 1 == 6 and sg == 6:
        p_set_after_break = prob_win_tiebreak(p_return_serve, p_serve, 0, 0)
        p_set_after_break = 1.0 - p_set_after_break
    else:
        p_set_after_break = 1.0 - prob_win_set(p_return_serve, p_serve, rg + 1, sg)

    p_set = g * p_set_after_hold + (1 - g) * p_set_after_break

    # Layer 4: probability server wins match from current set score
    sets_to_win = 2 if state.best_of == 3 else 3

    # Determine server's set count and receiver's set count
    if state.server == "player1":
        server_sets, receiver_sets = state.set_score
    else:
        receiver_sets, server_sets = state.set_score

    # Combine current set probability with remaining sets
    if server_sets + 1 >= sets_to_win:
        p_match_after_win_set = 1.0
    else:
        p_match_after_win_set = prob_win_match(
            p_serve, p_return_serve,
            server_sets + 1, receiver_sets, sets_to_win
        )

    if receiver_sets + 1 >= sets_to_win:
        p_match_after_lose_set = 0.0
    else:
        p_match_after_lose_set = prob_win_match(
            p_serve, p_return_serve,
            server_sets, receiver_sets + 1, sets_to_win
        )

    p_match = p_set * p_match_after_win_set + (1 - p_set) * p_match_after_lose_set

    # Clamp to avoid division by zero
    p_match = max(0.001, min(0.999, p_match))
    model_odds = 1.0 / p_match
    model_odds = nearest_tick(model_odds)

    return p_match, model_odds


def calculate_player1_win_prob(state: ScoreState) -> tuple[float, float]:
    """Return (probability_player1_wins, model_odds_for_player1).

    Internally calls calculate_model_odds which gives server's perspective,
    then converts to player1's perspective.
    """
    p_server, model_odds_server = calculate_model_odds(state)

    if state.server == "player1":
        p1 = p_server
    else:
        p1 = 1.0 - p_server

    p1 = max(0.001, min(0.999, p1))
    odds_p1 = nearest_tick(1.0 / p1)
    return p1, odds_p1
