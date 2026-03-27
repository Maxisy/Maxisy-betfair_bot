#!/usr/bin/env python3
"""Post-match backtester — merges live model tape with real Betfair historical data.

During a live match, the bot saves model_tape.jsonl: what the model calculated
at each second (model odds, score state, timestamps).

After the match, download the Betfair historical data file for that market.
This script merges the two: for each model snapshot, it finds the REAL
(non-delayed) Betfair odds at that timestamp and checks for edges.

Then it simulates the full trading logic: entry when edge > 6%, exit on
3-tick target / 4-tick stop loss / 60s timeout / edge gone.

Usage:
    python3 backtest_match.py data/historical/1.234567.bz2
    python3 backtest_match.py data/historical/1.234567.bz2 --market 1.234567
    python3 backtest_match.py data/historical/ --all

How to get Betfair historical data:
    1. Go to https://historicdata.betfair.com
    2. Login with your Betfair account
    3. Select Tennis > date > download the .bz2 or .tar.bz2 file
    4. Place in data/historical/
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

MODEL_TAPE_FILE = Path("data/model_tape.jsonl")

# Trading parameters (mirror config.py)
MIN_EDGE = 0.06         # 6%
MIN_ODDS = 1.15
MAX_ODDS = 4.00
STOP_LOSS_TICKS = 4
TARGET_PROFIT_TICKS = 3
MAX_HOLD_SECONDS = 60
EDGE_GONE_THRESHOLD = 0.02
COMMISSION_RATE = 0.05
STAKE = 10.00           # reference stake for P&L calculation
MIN_NET_PROFIT = 0.00   # no minimum for backtesting — we want to see all signals


# -----------------------------------------------------------------------
# Betfair tick ladder (copied from bot/ticks.py for standalone use)
# -----------------------------------------------------------------------

def _build_ladder() -> list[float]:
    bands = [
        (1.01, 2.00, 0.01), (2.00, 3.00, 0.02), (3.00, 4.00, 0.05),
        (4.00, 6.00, 0.10), (6.00, 10.00, 0.20), (10.00, 20.00, 0.50),
        (20.00, 30.00, 1.00),
    ]
    ladder: list[float] = []
    for lo, hi, step in bands:
        price = lo
        while price < hi - step / 2:
            ladder.append(round(price, 2))
            price += step
    ladder.append(30.0)
    return ladder

LADDER = _build_ladder()

import bisect

def tick_index(price: float) -> int:
    idx = bisect.bisect_left(LADDER, price - 1e-9)
    if idx < len(LADDER) and abs(LADDER[idx] - price) < 1e-9:
        return idx
    return bisect.bisect_left(LADDER, price)

def ticks_between(a: float, b: float) -> int:
    return tick_index(b) - tick_index(a)


# -----------------------------------------------------------------------
# Parse Betfair historical data
# -----------------------------------------------------------------------

def parse_betfair_historical(path: Path) -> dict[str, list[dict]]:
    """Parse a Betfair historical .bz2 or .tar.bz2 file.

    Returns dict of market_id -> list of tick snapshots sorted by time.
    Each tick: {"ts": epoch_seconds, "runners": {sel_id: {"back": x, "lay": y}}}
    """
    markets: dict[str, list[dict]] = defaultdict(list)
    # Current best prices per market per runner (delta accumulator)
    state: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"back": 0, "lay": 0}))

    lines = _read_lines(path)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        op = msg.get("op", "")
        if op != "mcm":
            continue

        pt = msg.get("pt", 0)  # publish time in epoch milliseconds
        ts = pt / 1000.0 if pt else 0

        for mc in msg.get("mc", []):
            market_id = mc.get("id", "")
            if not market_id:
                continue

            in_play = False
            md = mc.get("marketDefinition")
            if md:
                in_play = md.get("inPlay", False)

            for rc in mc.get("rc", []):
                sel_id = rc.get("id", 0)
                if sel_id == 0:
                    continue

                runner_state = state[market_id][sel_id]

                # Update best back (atb = available to back)
                atb = rc.get("atb", [])
                if atb:
                    # Best back = highest price available
                    best = max(atb, key=lambda x: x[0])
                    runner_state["back"] = best[0]

                # Update best lay (atl = available to lay)
                atl = rc.get("atl", [])
                if atl:
                    # Best lay = lowest price available
                    best = min(atl, key=lambda x: x[0])
                    runner_state["lay"] = best[0]

                # LTP fallback
                ltp = rc.get("ltp")
                if ltp and runner_state["back"] == 0:
                    runner_state["back"] = ltp
                if ltp and runner_state["lay"] == 0:
                    runner_state["lay"] = ltp

            # Save snapshot if in-play and we have timestamps
            if ts > 0 and in_play:
                snapshot = {
                    "ts": ts,
                    "runners": {
                        sel_id: {"back": s["back"], "lay": s["lay"]}
                        for sel_id, s in state[market_id].items()
                        if s["back"] > 0 or s["lay"] > 0
                    },
                }
                if snapshot["runners"]:
                    markets[market_id].append(snapshot)

    # Sort by timestamp
    for mid in markets:
        markets[mid].sort(key=lambda x: x["ts"])

    return dict(markets)


def _read_lines(path: Path) -> list[str]:
    """Read lines from .bz2, .tar.bz2, or plain JSON file."""
    name = path.name.lower()

    if name.endswith(".tar.bz2") or name.endswith(".tar"):
        lines = []
        with tarfile.open(str(path), "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        content = f.read()
                        try:
                            content = bz2.decompress(content)
                        except Exception:
                            pass
                        lines.extend(content.decode("utf-8", errors="replace").splitlines())
        return lines

    if name.endswith(".bz2"):
        with bz2.open(str(path), "rt", encoding="utf-8") as f:
            return f.readlines()

    # Plain JSON file
    with open(path, "r") as f:
        return f.readlines()


# -----------------------------------------------------------------------
# Load model tape
# -----------------------------------------------------------------------

def load_model_tape(market_id: str | None = None) -> dict[str, list[dict]]:
    """Load model_tape.jsonl, indexed by market_id."""
    if not MODEL_TAPE_FILE.exists():
        print(f"ERROR: {MODEL_TAPE_FILE} not found. Run the bot in paper mode first.")
        sys.exit(1)

    tape: dict[str, list[dict]] = defaultdict(list)
    with open(MODEL_TAPE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            mid = entry.get("market_id", "")
            if market_id and mid != market_id:
                continue
            tape[mid].append(entry)

    for mid in tape:
        tape[mid].sort(key=lambda x: x["unix_ts"])

    return dict(tape)


# -----------------------------------------------------------------------
# Find real odds at a given timestamp
# -----------------------------------------------------------------------

def find_real_odds(
    bf_tape: list[dict],
    target_ts: float,
    selection_id: int,
) -> dict[str, float] | None:
    """Binary search for the closest Betfair tick to target_ts."""
    if not bf_tape:
        return None

    # Binary search
    lo, hi = 0, len(bf_tape) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bf_tape[mid]["ts"] < target_ts:
            lo = mid + 1
        else:
            hi = mid

    # Check neighbours for closest
    best_idx = lo
    best_diff = abs(bf_tape[lo]["ts"] - target_ts)

    if lo > 0:
        diff = abs(bf_tape[lo - 1]["ts"] - target_ts)
        if diff < best_diff:
            best_idx = lo - 1
            best_diff = diff

    if best_diff > 5.0:  # no data within 5 seconds
        return None

    runners = bf_tape[best_idx].get("runners", {})
    sel_key = selection_id
    # Try both int and str keys
    runner = runners.get(sel_key) or runners.get(str(sel_key))
    return runner


# -----------------------------------------------------------------------
# Trade simulator
# -----------------------------------------------------------------------

def simulate_trades(
    model_tape: list[dict],
    bf_tape: list[dict],
) -> list[dict]:
    """Simulate trades using model decisions + real Betfair odds.

    For each model tick:
    1. Look up real Betfair odds at that timestamp
    2. Check if edge exists with real odds
    3. If entry signal: simulate entry and track position
    4. Check exit conditions every tick using real odds
    """
    trades: list[dict] = []
    position: dict | None = None

    for model_entry in model_tape:
        ts = model_entry["unix_ts"]
        sel_id = model_entry["selection_id"]
        model_odds = model_entry["model_odds_p1"]

        if model_odds <= 0:
            continue

        # Get real Betfair odds at this timestamp
        real = find_real_odds(bf_tape, ts, sel_id)
        if real is None:
            continue

        real_back = real.get("back", 0)
        real_lay = real.get("lay", 0)

        if real_back <= 0 or real_lay <= 0:
            continue

        # --- Position monitoring ---
        if position is not None:
            hold_time = ts - position["entry_ts"]

            # Current exit price (opposite side)
            if position["side"] == "BACK":
                exit_price = real_lay  # lay to close
            else:
                exit_price = real_back  # back to close

            # Check exit conditions
            exit_reason = None

            # Stop loss: 4 ticks against
            # BACK: we profit when odds DROP (lay cheaper). Against = odds RISE.
            # LAY: we profit when odds RISE (back cheaper). Against = odds DROP.
            if position["side"] == "BACK":
                ticks_against = max(0, ticks_between(position["entry_price"], exit_price))
                ticks_for = max(0, ticks_between(exit_price, position["entry_price"]))
            else:
                ticks_against = max(0, ticks_between(exit_price, position["entry_price"]))
                ticks_for = max(0, ticks_between(position["entry_price"], exit_price))

            if ticks_against >= STOP_LOSS_TICKS:
                exit_reason = "stop_loss"

            if not exit_reason and ticks_for >= TARGET_PROFIT_TICKS:
                exit_reason = "target_reached"

            # Timeout
            if not exit_reason and hold_time >= MAX_HOLD_SECONDS:
                exit_reason = "time_exit"

            # Edge gone
            if not exit_reason:
                current_edge = abs(real_back - model_odds) / model_odds
                if current_edge < EDGE_GONE_THRESHOLD:
                    exit_reason = "edge_gone"

            if exit_reason:
                # Close position
                # BACK at entry, LAY to close: profit when exit < entry (odds dropped)
                # LAY at entry, BACK to close: profit when exit > entry (odds rose)
                if position["side"] == "BACK":
                    gross = STAKE * (position["entry_price"] - exit_price) / position["entry_price"]
                else:
                    gross = STAKE * (exit_price - position["entry_price"]) / position["entry_price"]

                commission = max(0, gross) * COMMISSION_RATE
                net = gross - commission

                trade = {
                    "entry_ts": position["entry_ts"],
                    "exit_ts": ts,
                    "side": position["side"],
                    "entry_price": position["entry_price"],
                    "exit_price": round(exit_price, 2),
                    "model_odds": position["model_odds"],
                    "edge_pct": position["edge_pct"],
                    "hold_seconds": round(hold_time, 1),
                    "exit_reason": exit_reason,
                    "gross_pnl": round(gross, 4),
                    "commission": round(commission, 4),
                    "net_pnl": round(net, 4),
                    "stake": STAKE,
                    "score_at_entry": position.get("score", ""),
                    "player1": position.get("player1", ""),
                    "player2": position.get("player2", ""),
                    "tournament": position.get("tournament", ""),
                }
                trades.append(trade)
                position = None

            continue  # don't open new position while in one

        # --- Entry evaluation ---
        # Odds range check
        if real_back < MIN_ODDS or real_back > MAX_ODDS:
            continue

        # Edge calculation using REAL odds
        edge = abs(real_back - model_odds) / model_odds
        if edge < MIN_EDGE:
            continue

        # Determine side
        if real_back > model_odds:
            side = "BACK"
            entry_price = real_back
        else:
            side = "LAY"
            entry_price = real_lay

        # Check minimum profit
        gross = STAKE * abs(entry_price - model_odds) / model_odds
        commission = gross * COMMISSION_RATE
        net = gross - commission
        if net < MIN_NET_PROFIT:
            continue

        # Open position
        position = {
            "entry_ts": ts,
            "entry_price": entry_price,
            "side": side,
            "model_odds": model_odds,
            "edge_pct": round(edge * 100, 2),
            "score": f"P{model_entry.get('point_score', [])} G{model_entry.get('game_score', [])} S{model_entry.get('set_score', [])}",
            "player1": model_entry.get("player1", ""),
            "player2": model_entry.get("player2", ""),
            "tournament": model_entry.get("tournament", ""),
        }

    # Close any open position at end of data
    if position is not None:
        trades.append({
            "entry_ts": position["entry_ts"],
            "exit_ts": model_tape[-1]["unix_ts"] if model_tape else 0,
            "side": position["side"],
            "entry_price": position["entry_price"],
            "exit_price": position["entry_price"],  # flat
            "model_odds": position["model_odds"],
            "edge_pct": position["edge_pct"],
            "hold_seconds": 0,
            "exit_reason": "data_end",
            "gross_pnl": 0, "commission": 0, "net_pnl": 0,
            "stake": STAKE,
            "score_at_entry": position.get("score", ""),
            "player1": position.get("player1", ""),
            "player2": position.get("player2", ""),
            "tournament": position.get("tournament", ""),
        })

    return trades


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

def print_report(trades: list[dict], market_id: str) -> None:
    if not trades:
        print(f"\nNo trades generated for market {market_id}")
        return

    print(f"\n{'=' * 80}")
    print(f"BACKTEST RESULTS — Market {market_id}")
    print(f"{'=' * 80}")

    player_info = trades[0].get("player1", "")
    if player_info:
        print(f"  {trades[0].get('player1', '')} v {trades[0].get('player2', '')}")
        print(f"  {trades[0].get('tournament', '')}")

    print(f"\n{'#':>3}  {'Side':>4}  {'Entry':>6}  {'Exit':>6}  {'Model':>6}  "
          f"{'Edge%':>6}  {'Hold':>5}  {'Reason':>15}  {'Net P&L':>8}")
    print("-" * 80)

    total_pnl = 0
    wins = 0

    for i, t in enumerate(trades, 1):
        net = t["net_pnl"]
        total_pnl += net
        if net > 0:
            wins += 1

        print(f"{i:>3}  {t['side']:>4}  {t['entry_price']:>6.2f}  {t['exit_price']:>6.2f}  "
              f"{t['model_odds']:>6.2f}  {t['edge_pct']:>5.1f}%  "
              f"{t['hold_seconds']:>4.0f}s  {t['exit_reason']:>15}  "
              f"{'£' + f'{net:+.2f}':>8}")

    n = len(trades)
    wr = wins / n * 100 if n else 0
    avg_hold = sum(t["hold_seconds"] for t in trades) / n if n else 0
    total_commission = sum(t["commission"] for t in trades)

    print("-" * 80)
    print(f"\n  Total trades:    {n}")
    print(f"  Win rate:        {wr:.1f}%")
    print(f"  Gross P&L:       £{sum(t['gross_pnl'] for t in trades):+.2f}")
    print(f"  Commission:      £{total_commission:.2f}")
    print(f"  Net P&L:         £{total_pnl:+.2f}")
    print(f"  Avg hold time:   {avg_hold:.1f}s")

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in trades:
        r = t["exit_reason"]
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  Exit reasons:")
    for r, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {r:>15}: {count}")

    # Edge bucket analysis
    print(f"\n  By edge size:")
    for lo, hi in [(6, 8), (8, 10), (10, 15), (15, 100)]:
        bucket = [t for t in trades if lo <= t["edge_pct"] < hi]
        if bucket:
            bwr = sum(1 for t in bucket if t["net_pnl"] > 0) / len(bucket) * 100
            bavg = sum(t["net_pnl"] for t in bucket) / len(bucket)
            label = f"{lo}-{hi}%" if hi < 100 else f"{lo}%+"
            print(f"    {label:>8}: {len(bucket)} trades, {bwr:.0f}% win, avg £{bavg:+.3f}")

    # Verdict
    print(f"\n{'=' * 80}")
    if n >= 5 and wr > 55 and total_pnl > 0:
        print("VERDICT: Edge confirmed with real odds. Strategy looks viable.")
    elif n >= 5 and wr > 50:
        print("VERDICT: Marginal results. More matches needed.")
    elif n < 5:
        print("VERDICT: Too few trades to judge. Collect more data.")
    else:
        print("VERDICT: No edge with real odds. Review parameters.")
    print("=" * 80)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest model tape against real Betfair historical data"
    )
    parser.add_argument("historical_path", type=str,
                        help="Path to Betfair .bz2 file or directory of .bz2 files")
    parser.add_argument("--market", type=str, help="Specific market ID to backtest")
    parser.add_argument("--all", action="store_true",
                        help="Process all .bz2 files in directory")
    args = parser.parse_args()

    hist_path = Path(args.historical_path)

    # Collect files to process
    if hist_path.is_dir():
        files = sorted(hist_path.glob("*.bz2"))
        if not files:
            files = sorted(hist_path.glob("**/*.bz2"))
        if not files:
            print(f"No .bz2 files found in {hist_path}")
            sys.exit(1)
    else:
        files = [hist_path]

    # Load model tape
    model_tape = load_model_tape(args.market)
    print(f"Model tape: {sum(len(v) for v in model_tape.values())} entries "
          f"across {len(model_tape)} markets")

    all_trades: list[dict] = []

    for f in files:
        print(f"\nParsing {f.name}...")
        bf_markets = parse_betfair_historical(f)
        print(f"  Found {len(bf_markets)} markets, "
              f"{sum(len(v) for v in bf_markets.values())} ticks")

        for market_id, bf_tape in bf_markets.items():
            if args.market and market_id != args.market:
                continue

            if market_id not in model_tape:
                continue

            print(f"\n  Backtesting market {market_id}...")
            print(f"    Model ticks: {len(model_tape[market_id])}")
            print(f"    Betfair ticks: {len(bf_tape)}")

            trades = simulate_trades(model_tape[market_id], bf_tape)
            all_trades.extend(trades)
            print_report(trades, market_id)

    # Overall summary if multiple markets
    if len(all_trades) > 0 and len(set(t.get("tournament", "") for t in all_trades)) > 0:
        print(f"\n\n{'=' * 80}")
        print(f"OVERALL SUMMARY — {len(all_trades)} trades across all markets")
        print(f"{'=' * 80}")
        total_pnl = sum(t["net_pnl"] for t in all_trades)
        wins = sum(1 for t in all_trades if t["net_pnl"] > 0)
        n = len(all_trades)
        wr = wins / n * 100 if n else 0
        print(f"  Win rate:  {wr:.1f}%")
        print(f"  Net P&L:   £{total_pnl:+.2f}")
        print(f"  Per trade: £{total_pnl / n:+.3f}" if n else "")
        print("=" * 80)


if __name__ == "__main__":
    main()
