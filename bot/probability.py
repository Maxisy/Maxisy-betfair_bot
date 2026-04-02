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

    # Safety: game score beyond 7 means Goalserve is mid-update
    # (e.g., 7-6 before set score increments). Treat as set won/lost.
    if server_games >= 7:
        return 1.0
    if receiver_games >= 7:
        return 0.0

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

def opponent_adjusted_serve_pct(
    server_serve_pct: float,
    returner_return_pct: float,
    surface: str,
) -> float:
    """Adjust serve % based on opponent's return ability.

    Baseline: an "average" returner against this server would win
    (1 - server_serve_pct) of return points. If this specific returner
    is better/worse than that, adjust accordingly.

    This works across ATP/WTA/ITF because the baseline is relative to
    the server's own level, not a fixed tour average.
    """
    if returner_return_pct < 0.01 or server_serve_pct < 0.01:
        return server_serve_pct  # no return data, use raw serve %

    # What an "average" opponent returns against this server
    avg_ret_vs_server = 1.0 - server_serve_pct

    # How much better/worse is this returner vs average?
    ret_diff = returner_return_pct - avg_ret_vs_server

    # Dampen the adjustment — don't apply full difference, use 50%
    # to avoid over-correction from noisy data
    adjusted = server_serve_pct - ret_diff * 0.5

    return max(0.35, min(0.80, adjusted))


def adjusted_serve_pct(
    season_pct: float,
    service_games: int,
    service_holds: int,
    aces: int,
    double_faults: int,
    recent_holds: list[bool] | None = None,
) -> float:
    """Blend season serve % with in-match performance.

    Uses service hold rate and ace/DF counts to adjust the season average.
    Applies momentum weighting from recent service games.
    Small sample sizes are dampened to avoid overreacting early in a match.
    """
    if season_pct < 0.01 or service_games < 3:
        return season_pct  # not enough match data to adjust

    # Expected hold rate from season serve %
    expected_hold = prob_win_game(round(season_pct, 4), 0, 0)
    if expected_hold < 0.01:
        return season_pct

    # --- Momentum: weight recent games more heavily ---
    # If we have a rolling window, blend recent hold rate (60%) with
    # overall hold rate (40%) for the actual_hold calculation
    overall_hold = service_holds / service_games
    if recent_holds and len(recent_holds) >= 3:
        recent_hold_rate = sum(recent_holds) / len(recent_holds)
        actual_hold = 0.6 * recent_hold_rate + 0.4 * overall_hold
    else:
        actual_hold = overall_hold

    hold_ratio = actual_hold / expected_hold

    # Ace/DF signal: net aces per service game as a small bonus/penalty
    # Each net ace per game ≈ +0.5% serve, each net DF ≈ -0.5%
    net_aces = aces - double_faults
    ace_adj = (net_aces / service_games) * 0.005 if service_games > 0 else 0.0

    # Weight increases with sample size, caps at 0.3 (30% influence)
    weight = min(service_games / 20.0, 0.30)

    adjusted = season_pct * (1.0 + weight * (hold_ratio - 1.0)) + ace_adj
    return max(0.35, min(0.80, adjusted))


def fatigue_adjustment(serve_pct: float, total_points: int) -> float:
    """Apply fatigue penalty for long matches.

    After 200 points (~2 hours), serve % drops gradually.
    Penalty: -0.3% per 50 points beyond 200 (caps at -3%).
    """
    if total_points <= 200:
        return serve_pct

    excess = total_points - 200
    penalty = min(excess / 50 * 0.003, 0.03)  # max 3% penalty
    return max(0.35, serve_pct - penalty)


def set_context_adjustment(serve_pct: float, server_sets: int, receiver_sets: int,
                           sets_to_win: int) -> float:
    """Adjust serve % based on set score context.

    Trailing players serve more aggressively in deciding sets.
    Leading players play more conservatively.
    """
    if sets_to_win <= 1:
        return serve_pct  # not applicable

    # Deciding set (e.g., 1-1 in best of 3, 2-2 in best of 5)
    if server_sets == sets_to_win - 1 and receiver_sets == sets_to_win - 1:
        return serve_pct + 0.008  # +0.8% — both under pressure, server benefits

    # Trailing player: more aggressive serving
    if receiver_sets == sets_to_win - 1 and server_sets < receiver_sets:
        return serve_pct + 0.010  # +1.0% — desperation serving

    # Comfortably leading: slight conservatism
    if server_sets == sets_to_win - 1 and receiver_sets == 0:
        return serve_pct - 0.005  # -0.5% — may ease off

    return serve_pct


def elo_win_probability(elo1: float, elo2: float) -> float:
    """Expected win probability for player 1 from Elo ratings.

    Standard Elo formula: P(win) = 1 / (1 + 10^((elo2 - elo1) / 400))
    A 100-point gap ≈ 64% win rate (best-of-3 adjusted by TA).
    """
    if elo1 < 1000 or elo2 < 1000:
        return 0.5  # no valid Elo data
    return 1.0 / (1.0 + 10.0 ** ((elo2 - elo1) / 400.0))


def book_odds_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability (no margin removal)."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def calibrate_serve_pcts(
    p1_serve: float,
    p2_serve: float,
    target_p1_prob: float,
    best_of: int = 3,
) -> tuple[float, float]:
    """Calibrate serve %s so the Markov model matches a target win probability.

    Uses binary search on a quality offset δ applied symmetrically:
      p1_calibrated = p1_serve + δ
      p2_calibrated = p2_serve - δ

    This shifts the model's baseline to match book/Elo pre-match pricing,
    so in-match adjustments work relative to a correct starting point.

    Returns (p1_calibrated, p2_calibrated).
    """
    if target_p1_prob <= 0.01 or target_p1_prob >= 0.99:
        return p1_serve, p2_serve

    sets_to_win = 2 if best_of == 3 else 3

    def model_p1(delta: float) -> float:
        p1 = max(0.35, min(0.80, p1_serve + delta))
        p2 = max(0.35, min(0.80, p2_serve - delta))
        # Match probability from 0-0, 0-0 with P1 serving first
        return prob_win_match(round(p1, 4), round(p2, 4), 0, 0, sets_to_win)

    # Binary search on delta
    lo, hi = -0.20, 0.20
    for _ in range(40):
        mid = (lo + hi) / 2
        if model_p1(mid) < target_p1_prob:
            lo = mid
        else:
            hi = mid

    delta = (lo + hi) / 2
    return (
        max(0.35, min(0.80, p1_serve + delta)),
        max(0.35, min(0.80, p2_serve - delta)),
    )


def get_prior_p1_prob(
    opening_book_odds: tuple[float, float],
    elo1: float,
    elo2: float,
) -> float | None:
    """Get pre-match P1 win probability from book odds or Elo.

    Returns None if no prior available.
    """
    book_p1, book_p2 = opening_book_odds
    if book_p1 > 1.0 and book_p2 > 1.0:
        raw_p1 = book_odds_to_prob(book_p1)
        raw_p2 = book_odds_to_prob(book_p2)
        total = raw_p1 + raw_p2
        return raw_p1 / total if total > 0 else None

    if elo1 > 1000 and elo2 > 1000:
        return elo_win_probability(elo1, elo2)

    return None


def break_point_adjustment(serve_pct: float, bp_faced: int, bp_saved: int) -> float:
    """Adjust serve % based on break point performance.

    Players who save break points at a high rate are "clutch" servers.
    Players who crumble on break points will underperform their hold rate.
    Requires 4+ break points faced for signal.
    """
    if bp_faced < 4:
        return serve_pct

    save_rate = bp_saved / bp_faced
    # Expected save rate under pressure ≈ 60% (league average)
    expected_save = 0.60
    diff = save_rate - expected_save

    # Dampen: apply 40% of the difference as adjustment
    # Scale: 10% better save rate → +0.4% serve adjustment
    adj = diff * 0.04
    return max(0.35, min(0.80, serve_pct + adj))


def calculate_model_odds(state: ScoreState) -> tuple[float, float]:
    """Calculate model probability and decimal odds for the SERVER winning.

    Pipeline:
    1. Get raw serve %s from TA + opponent return adjustment
    2. Calibrate against book odds / Elo (shift serve %s to match pre-match pricing)
    3. Apply in-match adjustments (holds/breaks/momentum/aces)
    4. Apply fatigue, set context, break point adjustments
    5. Run Markov chain from current score state

    Returns (probability, model_odds_decimal).
    model_odds is rounded to nearest Betfair tick.
    """
    # Step 1: Adjust serve % for opponent's return ability
    p1_serve_raw = opponent_adjusted_serve_pct(
        state.player1_serve_pct, state.player2_return_pct, state.surface)
    p2_serve_raw = opponent_adjusted_serve_pct(
        state.player2_serve_pct, state.player1_return_pct, state.surface)

    # Step 2: Calibrate serve %s against book odds / Elo prior
    # This shifts the model inputs so that at 0-0 0-0, the model produces
    # the same probability as the bookmaker. Quality gaps are now baked in.
    prior = get_prior_p1_prob(
        state.opening_book_odds, state.player1_elo, state.player2_elo)
    if prior is not None:
        p1_base, p2_base = calibrate_serve_pcts(
            p1_serve_raw, p2_serve_raw, prior, state.best_of)
    else:
        p1_base, p2_base = p1_serve_raw, p2_serve_raw

    # Step 3: In-match adjustments (holds/breaks/aces + momentum)
    # These adjust RELATIVE to the calibrated baseline
    if state.server == "player1":
        p_serve = adjusted_serve_pct(
            p1_base,
            state.player1_service_games,
            state.player1_service_holds,
            state.player1_aces,
            state.player1_double_faults,
            state.player1_recent_holds,
        )
        p_return_serve = adjusted_serve_pct(
            p2_base,
            state.player2_service_games,
            state.player2_service_holds,
            state.player2_aces,
            state.player2_double_faults,
            state.player2_recent_holds,
        )
        server_sets, receiver_sets = state.set_score
        server_bp_faced = state.player1_bp_faced
        server_bp_saved = state.player1_bp_saved
    else:
        p_serve = adjusted_serve_pct(
            p2_base,
            state.player2_service_games,
            state.player2_service_holds,
            state.player2_aces,
            state.player2_double_faults,
            state.player2_recent_holds,
        )
        p_return_serve = adjusted_serve_pct(
            p1_base,
            state.player1_service_games,
            state.player1_service_holds,
            state.player1_aces,
            state.player1_double_faults,
            state.player1_recent_holds,
        )
        receiver_sets, server_sets = state.set_score
        server_bp_faced = state.player2_bp_faced
        server_bp_saved = state.player2_bp_saved

    # Step 4: Fatigue adjustment for long matches
    p_serve = fatigue_adjustment(p_serve, state.total_points_played)
    p_return_serve = fatigue_adjustment(p_return_serve, state.total_points_played)

    # Step 5: Set context adjustment (desperation/comfort serving)
    sets_to_win = 2 if state.best_of == 3 else 3
    p_serve = set_context_adjustment(p_serve, server_sets, receiver_sets, sets_to_win)
    p_return_serve = set_context_adjustment(p_return_serve, receiver_sets, server_sets, sets_to_win)

    # Step 6: Break point clutch/choke adjustment
    p_serve = break_point_adjustment(p_serve, server_bp_faced, server_bp_saved)

    sg, rg = state.game_score

    if state.is_tiebreak:
        # In a tiebreak: point_score is actual tiebreak points (0,1,2,...),
        # and game_score should be 6-6. Use tiebreak probability directly.
        g = prob_win_tiebreak(
            p_serve, p_return_serve,
            state.point_score[0], state.point_score[1],
        )
        # Tiebreak winner takes the set
        p_set = g
    else:
        # Layer 2: probability server wins current game
        g = prob_win_game(p_serve, state.point_score[0], state.point_score[1])

        # Layer 3: probability server wins current set from current game score
        # First get prob of winning set assuming current game is won or lost,
        # weighted by prob of winning current game.

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
    # (sets_to_win, server_sets, receiver_sets already computed in step 4)

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

    The model's serve %s are calibrated against book odds / Elo pre-match,
    so the Markov chain starts from correct baseline probabilities.
    In-match adjustments then shift relative to that baseline.
    """
    p_server, model_odds_server = calculate_model_odds(state)

    if state.server == "player1":
        p1 = p_server
    else:
        p1 = 1.0 - p_server

    p1 = max(0.001, min(0.999, p1))
    odds_p1 = nearest_tick(1.0 / p1)
    return p1, odds_p1


def calculate_player1_win_prob_uncalibrated(state: ScoreState) -> tuple[float, float]:
    """Return model probability WITHOUT book/Elo calibration (for comparison).

    Uses raw TA serve %s + opponent adjustment only, no book odds calibration.
    Useful for logging to see how much the calibration changed things.
    """
    # Temporarily clear the prior sources
    saved_book = state.opening_book_odds
    saved_elo1 = state.player1_elo
    saved_elo2 = state.player2_elo
    state.opening_book_odds = (0.0, 0.0)
    state.player1_elo = 0.0
    state.player2_elo = 0.0

    try:
        p_server, _ = calculate_model_odds(state)
        if state.server == "player1":
            p1 = p_server
        else:
            p1 = 1.0 - p_server
        p1 = max(0.001, min(0.999, p1))
        odds_p1 = nearest_tick(1.0 / p1)
        return p1, odds_p1
    finally:
        state.opening_book_odds = saved_book
        state.player1_elo = saved_elo1
        state.player2_elo = saved_elo2
