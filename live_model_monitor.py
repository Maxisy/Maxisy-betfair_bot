"""Live model vs book odds monitor — runs 24/7 logging all data.

Polls Goalserve every 1s (like real bot), builds in-match hold/break
history, compares model odds vs book odds, and logs everything to JSONL.

Usage: python3 live_model_monitor.py
Output: data/model_monitor.jsonl (one line per match per snapshot)
        data/model_monitor_summary.jsonl (periodic aggregate stats)

Ctrl+C to stop.
"""

import asyncio
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, ".")
from bot.config import Config
from bot.goalserve import GoalservePoller, INPLAY_URL
from bot.serve_stats import ServeStatsLoader
from bot.probability import (
    calculate_player1_win_prob, calculate_player1_win_prob_uncalibrated,
    adjusted_serve_pct, opponent_adjusted_serve_pct,
    fatigue_adjustment, set_context_adjustment, break_point_adjustment,
    elo_win_probability, get_prior_p1_prob, calibrate_serve_pcts,
)

config = Config()

LOG_FILE = Path("data/model_monitor.jsonl")
SUMMARY_FILE = Path("data/model_monitor_summary.jsonl")
SNAPSHOT_INTERVAL = 30  # log snapshot every 30s
SUMMARY_INTERVAL = 300  # print summary every 5 min
FETCH_DELAY = 2.0  # delay between TA fetches

# Track matches we've already fetched TA data for
_fetched_players: set[tuple[str, str]] = set()


def is_allowed(tournament: str) -> bool:
    t = tournament.lower()
    for ex in config.excluded_tournaments:
        if ex.lower() in t:
            return False
    for al in config.allowed_tournaments:
        if al.lower() in t:
            return True
    return False


def parse_book_odds(raw_data: dict) -> dict[str, tuple[float, float]]:
    """Extract book odds from Goalserve feed (market 67 = To Win)."""
    book_odds = {}
    events = raw_data.get("events") or {}
    if not isinstance(events, dict):
        return book_odds
    for _key, evt in events.items():
        if not isinstance(evt, dict):
            continue
        info = evt.get("info") or {}
        match_id = str(info.get("id", ""))
        odds = evt.get("odds") or {}
        to_win = odds.get("67") or {}
        participants = to_win.get("participants") or {}
        if not isinstance(participants, dict):
            continue
        p1_odds = None
        p2_odds = None
        for _pk, pv in participants.items():
            name = pv.get("name", "")
            try:
                val = float(pv.get("value_eu", 0))
            except (ValueError, TypeError):
                continue
            if name == "Home":
                p1_odds = val
            elif name == "Away":
                p2_odds = val
        if p1_odds and p2_odds:
            book_odds[match_id] = (p1_odds, p2_odds)
    return book_odds


async def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Clear previous session data
    if LOG_FILE.exists():
        LOG_FILE.unlink()
        print(f"Cleared previous {LOG_FILE}")
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()
        print(f"Cleared previous {SUMMARY_FILE}")

    # Set up components
    loader = ServeStatsLoader(config)
    loader._load_cached()
    loader._session = aiohttp.ClientSession()
    await loader._refresh_name_index()

    poller = GoalservePoller(config)
    session = aiohttp.ClientSession()

    shutdown = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown.set)

    print(f"Cache: {len(loader.stats)} players | Name index: {len(loader._name_index)} words")
    print(f"Logging to: {LOG_FILE}")
    print(f"Snapshot every {SNAPSHOT_INTERVAL}s | Summary every {SUMMARY_INTERVAL}s")
    print(f"Press Ctrl+C to stop.\n")

    last_snapshot = 0.0
    last_summary = 0.0
    poll_count = 0
    total_snapshots = 0
    all_edges: list[float] = []
    # Per-match tracking for edge over time
    match_history: dict[str, list[dict]] = {}

    while not shutdown.is_set():
        try:
            # Poll Goalserve
            async with session.get(
                INPLAY_URL, timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    print(f"[{_ts()}] Goalserve error: {resp.status}")
                    await asyncio.sleep(1)
                    continue
                raw_data = await resp.json(content_type=None)

            poller._parse_feed(raw_data)
            book_odds = parse_book_odds(raw_data)
            poll_count += 1

            # Fetch TA data for new players (non-blocking, with delay)
            for mid, state in poller.scores.items():
                if not is_allowed(state.tournament):
                    continue
                for name, surface in [
                    (state.player1_name, state.surface),
                    (state.player2_name, state.surface),
                ]:
                    if not name:
                        continue
                    key = (name.lower(), surface)
                    if key in _fetched_players:
                        continue
                    if loader.get_serve_pct(name, surface) is not None:
                        _fetched_players.add(key)
                        continue
                    # Fetch in background
                    _fetched_players.add(key)
                    asyncio.create_task(_fetch_player(loader, name, surface))

            # Set serve, return, and Elo stats on score states
            for mid, state in poller.scores.items():
                p1 = loader.get_serve_pct(state.player1_name, state.surface)
                p2 = loader.get_serve_pct(state.player2_name, state.surface)
                state.player1_serve_pct = p1 if p1 is not None else 0.0
                state.player2_serve_pct = p2 if p2 is not None else 0.0
                r1 = loader.get_return_pct(state.player1_name, state.surface)
                r2 = loader.get_return_pct(state.player2_name, state.surface)
                state.player1_return_pct = r1 if r1 is not None else 0.0
                state.player2_return_pct = r2 if r2 is not None else 0.0
                # Elo ratings
                e1 = loader.get_elo(state.player1_name, state.surface)
                e2 = loader.get_elo(state.player2_name, state.surface)
                state.player1_elo = e1 if e1 is not None else 0.0
                state.player2_elo = e2 if e2 is not None else 0.0
                # Capture opening book odds (first time we see them)
                if state.opening_book_odds == (0.0, 0.0):
                    bk = book_odds.get(mid)
                    if bk:
                        state.opening_book_odds = bk

            now = time.time()

            # Snapshot every SNAPSHOT_INTERVAL seconds
            if now - last_snapshot >= SNAPSHOT_INTERVAL:
                last_snapshot = now
                total_snapshots += 1
                snapshot_edges = []

                # Count skipped mid-match joins
                skipped_mid_match = 0

                for mid, state in sorted(
                    poller.scores.items(), key=lambda x: x[1].tournament,
                ):
                    if not is_allowed(state.tournament):
                        continue
                    if not state.server:
                        continue
                    if state.player1_serve_pct < 0.01 or state.player2_serve_pct < 0.01:
                        continue
                    if not state.tracked_from_start:
                        skipped_mid_match += 1
                        continue

                    bk = book_odds.get(mid)
                    if not bk:
                        continue

                    p1_prob, p1_model_odds = calculate_player1_win_prob(state)
                    p1_raw_prob, p1_raw_odds = calculate_player1_win_prob_uncalibrated(state)
                    p1_book, p2_book = bk
                    p2_model_odds = 1.0 / (1.0 - p1_prob) if p1_prob < 0.999 else 1000.0

                    edge_p1 = (1.0 / p1_book - p1_prob) * 100 if p1_book > 1 else 0
                    raw_edge_p1 = (1.0 / p1_book - p1_raw_prob) * 100 if p1_book > 1 else 0

                    # Recompute calibrated serve %s for logging
                    p1_opp_adj = opponent_adjusted_serve_pct(
                        state.player1_serve_pct, state.player2_return_pct, state.surface)
                    p2_opp_adj = opponent_adjusted_serve_pct(
                        state.player2_serve_pct, state.player1_return_pct, state.surface)
                    # Get calibrated base (same as model uses internally)
                    prior = get_prior_p1_prob(
                        state.opening_book_odds, state.player1_elo, state.player2_elo)
                    if prior is not None:
                        p1_cal, p2_cal = calibrate_serve_pcts(
                            p1_opp_adj, p2_opp_adj, prior, state.best_of)
                    else:
                        p1_cal, p2_cal = p1_opp_adj, p2_opp_adj
                    p1_adj = adjusted_serve_pct(
                        p1_cal,
                        state.player1_service_games,
                        state.player1_service_holds,
                        state.player1_aces,
                        state.player1_double_faults,
                        state.player1_recent_holds,
                    )
                    p2_adj = adjusted_serve_pct(
                        p2_cal,
                        state.player2_service_games,
                        state.player2_service_holds,
                        state.player2_aces,
                        state.player2_double_faults,
                        state.player2_recent_holds,
                    )

                    record = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "match_id": mid,
                        "tournament": state.tournament,
                        "surface": state.surface,
                        "p1": state.player1_name,
                        "p2": state.player2_name,
                        "server": state.server,
                        "set_score": list(state.set_score),
                        "game_score": list(state.game_score),
                        "point_score": list(state.point_score),
                        "is_tiebreak": state.is_tiebreak,
                        "p1_serve_season": round(state.player1_serve_pct, 4),
                        "p2_serve_season": round(state.player2_serve_pct, 4),
                        "p1_return_season": round(state.player1_return_pct, 4),
                        "p2_return_season": round(state.player2_return_pct, 4),
                        "p1_serve_adjusted": round(p1_adj, 4),
                        "p2_serve_adjusted": round(p2_adj, 4),
                        "p1_svc_games": state.player1_service_games,
                        "p1_svc_holds": state.player1_service_holds,
                        "p2_svc_games": state.player2_service_games,
                        "p2_svc_holds": state.player2_service_holds,
                        "p1_aces": state.player1_aces,
                        "p1_dfs": state.player1_double_faults,
                        "p2_aces": state.player2_aces,
                        "p2_dfs": state.player2_double_faults,
                        "p1_recent_holds": state.player1_recent_holds[-5:],
                        "p2_recent_holds": state.player2_recent_holds[-5:],
                        "total_points": state.total_points_played,
                        "p1_bp": f"{state.player1_bp_saved}/{state.player1_bp_faced}",
                        "p2_bp": f"{state.player2_bp_saved}/{state.player2_bp_faced}",
                        "p1_elo": round(state.player1_elo, 1),
                        "p2_elo": round(state.player2_elo, 1),
                        "p1_opening_book": state.opening_book_odds[0],
                        "p2_opening_book": state.opening_book_odds[1],
                        "p1_model_odds_blended": round(p1_model_odds, 3),
                        "p1_model_odds_raw": round(p1_raw_odds, 3),
                        "p2_model_odds": round(p2_model_odds, 3),
                        "p1_book_odds": p1_book,
                        "p2_book_odds": p2_book,
                        "p1_edge_pct": round(edge_p1, 2),
                        "p1_raw_edge_pct": round(raw_edge_p1, 2),
                        "p1_prob_blended": round(p1_prob, 4),
                        "p1_prob_raw": round(p1_raw_prob, 4),
                        "tracked_from_start": state.tracked_from_start,
                    }

                    # Write to log
                    with open(LOG_FILE, "a") as f:
                        f.write(json.dumps(record) + "\n")

                    snapshot_edges.append(abs(edge_p1))
                    all_edges.append(abs(edge_p1))

                    # Track per-match history
                    if mid not in match_history:
                        match_history[mid] = []
                    match_history[mid].append(record)

                if snapshot_edges:
                    avg_e = sum(snapshot_edges) / len(snapshot_edges)
                    max_e = max(snapshot_edges)
                    within_5 = sum(1 for e in snapshot_edges if e <= 5)
                    skip_str = f" | skipped {skipped_mid_match} mid-match" if skipped_mid_match else ""
                    print(
                        f"[{_ts()}] Poll #{poll_count:>5d} | "
                        f"{len(snapshot_edges):>2d} matches | "
                        f"avg |edge| {avg_e:>5.1f}% | "
                        f"max |edge| {max_e:>5.1f}% | "
                        f"within 5%: {within_5}/{len(snapshot_edges)}{skip_str}"
                    )
                else:
                    skip_str = f" | skipped {skipped_mid_match} mid-match" if skipped_mid_match else ""
                    print(f"[{_ts()}] Poll #{poll_count:>5d} | 0 tradeable matches{skip_str}")

            # Periodic summary
            if now - last_summary >= SUMMARY_INTERVAL:
                last_summary = now
                _print_summary(
                    poll_count, total_snapshots, all_edges,
                    match_history, poller, loader, book_odds,
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[{_ts()}] Error: {e}")

        await asyncio.sleep(1.0)

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    _print_summary(
        poll_count, total_snapshots, all_edges,
        match_history, poller, loader, book_odds,
    )

    # Save final summary
    if all_edges:
        summary = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_polls": poll_count,
            "total_snapshots": total_snapshots,
            "total_edge_samples": len(all_edges),
            "avg_abs_edge": round(sum(all_edges) / len(all_edges), 2),
            "max_abs_edge": round(max(all_edges), 2),
            "within_5pct": sum(1 for e in all_edges if e <= 5),
            "within_10pct": sum(1 for e in all_edges if e <= 10),
            "within_15pct": sum(1 for e in all_edges if e <= 15),
            "matches_tracked": len(match_history),
        }
        with open(SUMMARY_FILE, "a") as f:
            f.write(json.dumps(summary) + "\n")

    await session.close()
    await loader._session.close()
    loader._save_cache()
    print(f"\nLog saved to {LOG_FILE}")
    print(f"Summary saved to {SUMMARY_FILE}")


async def _fetch_player(loader: ServeStatsLoader, name: str, surface: str):
    """Fetch a player's TA data with delay to avoid rate limiting."""
    await asyncio.sleep(FETCH_DELAY)
    try:
        await loader._fetch_and_cache(name, surface)
        pct = loader.get_serve_pct(name, surface)
        if pct:
            print(f"[{_ts()}] Fetched: {name} [{surface}] = {pct:.1%}")
        else:
            # Distinguish "player exists but no surface data" from "not on TA"
            stats = loader._find_player_stats(name)
            if stats:
                available = [k for k in stats if not k.endswith("_ret")]
                print(f"[{_ts()}] No {surface} data: {name} (has: {', '.join(available)})")
            else:
                print(f"[{_ts()}] Not found on TA: {name}")
    except Exception as e:
        print(f"[{_ts()}] Fetch error for {name}: {e}")


def _print_summary(
    poll_count, total_snapshots, all_edges,
    match_history, poller, loader, book_odds,
):
    """Print detailed summary to stdout."""
    print(f"\n{'─'*80}")
    print(f"[{_ts()}] SUMMARY after {poll_count} polls, {total_snapshots} snapshots")
    print(f"{'─'*80}")

    if all_edges:
        avg_e = sum(all_edges) / len(all_edges)
        print(f"  Total edge samples: {len(all_edges)}")
        print(f"  Avg |edge|:  {avg_e:.1f}%")
        print(f"  Max |edge|:  {max(all_edges):.1f}%")
        print(f"  Within 5%:   {sum(1 for e in all_edges if e <= 5)}/{len(all_edges)}")
        print(f"  Within 10%:  {sum(1 for e in all_edges if e <= 10)}/{len(all_edges)}")
        print(f"  Within 15%:  {sum(1 for e in all_edges if e <= 15)}/{len(all_edges)}")

        # Edge distribution
        buckets = [0, 2, 5, 10, 15, 20, 50]
        print(f"\n  Edge distribution:")
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            count = sum(1 for e in all_edges if lo <= e < hi)
            bar = "█" * (count * 40 // len(all_edges)) if all_edges else ""
            print(f"    {lo:>2d}-{hi:>2d}%: {count:>4d} ({count*100/len(all_edges):>5.1f}%) {bar}")
        count = sum(1 for e in all_edges if e >= 50)
        if count:
            print(f"    50%+: {count:>4d} ({count*100/len(all_edges):>5.1f}%)")

    # Per-match summaries (matches with enough data)
    print(f"\n  Matches tracked: {len(match_history)}")
    for mid, history in sorted(match_history.items(), key=lambda x: -len(x[1])):
        if len(history) < 2:
            continue
        edges = [abs(h["p1_edge_pct"]) for h in history]
        last = history[-1]
        holds = f"P1:{last['p1_svc_holds']}/{last['p1_svc_games']} P2:{last['p2_svc_holds']}/{last['p2_svc_games']}"
        extra = f" pts:{last.get('total_points', '?')}"
        extra += f" bp:{last.get('p1_bp', '?')}/{last.get('p2_bp', '?')}"
        adj_info = ""
        if last["p1_serve_adjusted"] != last["p1_serve_season"]:
            adj_info = f" adj:{last['p1_serve_adjusted']:.1%}/{last['p2_serve_adjusted']:.1%}"
        print(
            f"    {last['p1']:20s} vs {last['p2']:20s} | "
            f"{len(history):>3d} pts | "
            f"avg edge {sum(edges)/len(edges):>5.1f}% | "
            f"{holds}{extra}{adj_info}"
        )

    print(f"{'─'*80}\n")


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


asyncio.run(main())
