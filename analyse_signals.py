#!/usr/bin/env python3
"""Analyse recorded signals against the odds tape.

For each signal, looks up what the market odds were at T+5s, T+10s, T+30s,
T+60s after the signal fired. Calculates whether the predicted reversion
actually happened and what the P&L would have been.

Usage:
    python3 analyse_signals.py                     # analyse all signals
    python3 analyse_signals.py --market 1.234567   # specific market only
    python3 analyse_signals.py --min-edge 8        # only signals with >= 8% edge
    python3 analyse_signals.py --csv                # output as CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SIGNALS_FILE = Path("data/signals.jsonl")
ODDS_TAPE_FILE = Path("data/odds_tape.jsonl")

# Check-back windows in seconds
WINDOWS = [5, 10, 15, 30, 60]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the bot in paper mode first.")
        sys.exit(1)
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_tape_index(tape: list[dict]) -> dict[str, list[dict]]:
    """Index tape entries by market_id, sorted by timestamp."""
    idx: dict[str, list[dict]] = defaultdict(list)
    for entry in tape:
        mid = entry.get("market_id", "")
        if mid:
            idx[mid].append(entry)
    for mid in idx:
        idx[mid].sort(key=lambda e: e["unix_ts"])
    return idx


def find_odds_at(tape_entries: list[dict], target_ts: float,
                  selection_id: str) -> dict | None:
    """Find the tape entry closest to target_ts (within 3s)."""
    best = None
    best_diff = float("inf")
    for entry in tape_entries:
        diff = abs(entry["unix_ts"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best = entry
        elif entry["unix_ts"] > target_ts + 5:
            break  # past the window, stop searching
    if best and best_diff <= 3.0:
        runners = best.get("runners", {})
        return runners.get(selection_id)
    return None


def analyse(signals: list[dict], tape_index: dict[str, list[dict]],
            args: argparse.Namespace) -> None:
    """Analyse each signal against the tape."""
    results = []

    for sig in signals:
        market_id = sig["market_id"]
        sel_id = str(sig["selection_id"])
        signal_ts = sig["unix_ts"]
        side = sig["side"]
        model_odds = sig["model_odds"]
        entry_back = sig["market_back"]
        entry_lay = sig["market_lay"]
        edge = sig["edge_pct"]

        # Filters
        if args.market and market_id != args.market:
            continue
        if edge < args.min_edge:
            continue

        tape = tape_index.get(market_id, [])
        if not tape:
            continue

        # Entry price
        entry_price = entry_back if side == "BACK" else entry_lay

        # Look up future odds at each window
        windows_data = {}
        for w in WINDOWS:
            runner = find_odds_at(tape, signal_ts + w, sel_id)
            if runner:
                future_back = runner.get("back", 0)
                future_lay = runner.get("lay", 0)
                # Exit price: opposite side
                exit_price = future_lay if side == "BACK" else future_back

                if entry_price > 0 and exit_price > 0:
                    # P&L calculation
                    if side == "BACK":
                        # Backed at entry_price, lay to close at exit_price
                        # Profit if exit > entry (price went up = bad for backer)
                        # Wait — for BACK, we want price to go UP (higher odds = overpriced, reversion = down)
                        # Actually: we BACKED because market > model. We expect market to drop toward model.
                        # So profit = (entry_price - exit_price) * stake / entry_price (approx)
                        tick_move = round((exit_price - entry_price) / 0.01)  # rough tick approximation
                        raw_pnl_pct = (entry_price - exit_price) / entry_price * 100
                    else:
                        # LAY: we laid because market < model. Expect market to rise.
                        tick_move = round((entry_price - exit_price) / 0.01)
                        raw_pnl_pct = (exit_price - entry_price) / entry_price * 100

                    windows_data[w] = {
                        "exit_back": round(future_back, 2),
                        "exit_lay": round(future_lay, 2),
                        "exit_price": round(exit_price, 2),
                        "raw_pnl_pct": round(raw_pnl_pct, 2),
                        "reverted": raw_pnl_pct > 0,
                    }

        if not windows_data:
            continue

        result = {
            "signal_id": sig["signal_id"],
            "timestamp": sig["timestamp"],
            "event_name": sig["event_name"],
            "tournament": sig.get("tournament", ""),
            "side": side,
            "model_odds": model_odds,
            "entry_back": entry_back,
            "entry_lay": entry_lay,
            "edge_pct": edge,
            "score": f"P{sig['point_score']} G{sig['game_score']} S{sig['set_score']}",
            "windows": windows_data,
        }
        results.append(result)

    # --- Output ---
    if not results:
        print("No signals found matching criteria.")
        print(f"  Signals file: {SIGNALS_FILE} ({SIGNALS_FILE.stat().st_size} bytes)")
        print(f"  Tape file: {ODDS_TAPE_FILE} ({ODDS_TAPE_FILE.stat().st_size} bytes)")
        return

    if args.csv:
        print_csv(results)
    else:
        print_report(results)


def print_report(results: list[dict]) -> None:
    """Print human-readable analysis report."""
    print("=" * 80)
    print(f"SIGNAL ANALYSIS — {len(results)} signals analysed")
    print("=" * 80)

    # Per-signal detail
    for r in results:
        print(f"\n--- Signal #{r['signal_id']} at {r['timestamp']} ---")
        print(f"  {r['event_name']} ({r['tournament']})")
        print(f"  {r['side']} | model={r['model_odds']:.2f} | "
              f"back={r['entry_back']:.2f} lay={r['entry_lay']:.2f} | "
              f"edge={r['edge_pct']:.1f}% | {r['score']}")

        for w in WINDOWS:
            if w in r["windows"]:
                wd = r["windows"][w]
                status = "PROFIT" if wd["reverted"] else "LOSS"
                print(f"  T+{w:2d}s: exit_price={wd['exit_price']:.2f} "
                      f"pnl={wd['raw_pnl_pct']:+.2f}%  [{status}]")
            else:
                print(f"  T+{w:2d}s: no data")

    # --- Summary stats per window ---
    print("\n" + "=" * 80)
    print("SUMMARY BY TIME WINDOW")
    print("=" * 80)
    print(f"{'Window':>8}  {'Signals':>8}  {'Reverted':>9}  {'Win Rate':>9}  "
          f"{'Avg P&L%':>9}  {'Med P&L%':>9}  {'Best':>8}  {'Worst':>8}")
    print("-" * 80)

    for w in WINDOWS:
        pnls = []
        reverted = 0
        for r in results:
            if w in r["windows"]:
                wd = r["windows"][w]
                pnls.append(wd["raw_pnl_pct"])
                if wd["reverted"]:
                    reverted += 1

        if not pnls:
            print(f"  T+{w:2d}s:  no data")
            continue

        pnls_sorted = sorted(pnls)
        n = len(pnls)
        avg = sum(pnls) / n
        med = pnls_sorted[n // 2]
        wr = reverted / n * 100

        print(f"  T+{w:2d}s  {n:>8}  {reverted:>9}  {wr:>8.1f}%  "
              f"{avg:>+8.2f}%  {med:>+8.2f}%  {max(pnls):>+7.2f}%  {min(pnls):>+7.2f}%")

    # --- Summary by edge bucket ---
    print("\n" + "=" * 80)
    print("SUMMARY BY EDGE SIZE (at T+30s)")
    print("=" * 80)

    buckets = [(6, 8), (8, 10), (10, 15), (15, 100)]
    print(f"{'Edge':>10}  {'Signals':>8}  {'Win Rate':>9}  {'Avg P&L%':>9}")
    print("-" * 45)

    for lo, hi in buckets:
        pnls = []
        for r in results:
            if lo <= r["edge_pct"] < hi and 30 in r["windows"]:
                pnls.append(r["windows"][30]["raw_pnl_pct"])
        if pnls:
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            label = f"{lo}-{hi}%" if hi < 100 else f"{lo}%+"
            print(f"  {label:>8}  {len(pnls):>8}  {wr:>8.1f}%  {avg:>+8.2f}%")

    # --- By side ---
    print("\n" + "=" * 80)
    print("SUMMARY BY SIDE (at T+30s)")
    print("=" * 80)

    for side in ("BACK", "LAY"):
        pnls = [r["windows"][30]["raw_pnl_pct"]
                for r in results if r["side"] == side and 30 in r["windows"]]
        if pnls:
            avg = sum(pnls) / len(pnls)
            wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
            print(f"  {side:>5}: {len(pnls)} signals, {wr:.1f}% win rate, avg {avg:+.2f}%")

    # --- Verdict ---
    print("\n" + "=" * 80)
    all_30s = [r["windows"][30]["raw_pnl_pct"] for r in results if 30 in r["windows"]]
    if all_30s:
        avg = sum(all_30s) / len(all_30s)
        wr = sum(1 for p in all_30s if p > 0) / len(all_30s) * 100
        print(f"OVERALL at T+30s: {len(all_30s)} signals, {wr:.1f}% win rate, avg {avg:+.2f}%")
        if wr > 55 and avg > 0:
            print("VERDICT: Positive edge detected. Consider live testing with micro-stakes.")
        elif wr > 50:
            print("VERDICT: Marginal edge. More data needed before committing capital.")
        else:
            print("VERDICT: No clear edge. Review model parameters or target markets.")
    print("=" * 80)


def print_csv(results: list[dict]) -> None:
    """Output as CSV for spreadsheet analysis."""
    headers = [
        "signal_id", "timestamp", "event_name", "tournament", "side",
        "model_odds", "entry_back", "entry_lay", "edge_pct", "score",
    ]
    for w in WINDOWS:
        headers.extend([f"exit_price_T{w}", f"pnl_pct_T{w}", f"reverted_T{w}"])

    print(",".join(headers))

    for r in results:
        row = [
            str(r["signal_id"]), r["timestamp"], f'"{r["event_name"]}"',
            f'"{r["tournament"]}"', r["side"],
            f'{r["model_odds"]:.4f}', f'{r["entry_back"]:.2f}',
            f'{r["entry_lay"]:.2f}', f'{r["edge_pct"]:.2f}', f'"{r["score"]}"',
        ]
        for w in WINDOWS:
            if w in r["windows"]:
                wd = r["windows"][w]
                row.extend([
                    f'{wd["exit_price"]:.2f}',
                    f'{wd["raw_pnl_pct"]:.2f}',
                    str(wd["reverted"]),
                ])
            else:
                row.extend(["", "", ""])
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse dry-run signals against odds tape")
    parser.add_argument("--market", type=str, help="Filter to specific market ID")
    parser.add_argument("--min-edge", type=float, default=0, help="Minimum edge %% to include")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    args = parser.parse_args()

    signals = load_jsonl(SIGNALS_FILE)
    tape = load_jsonl(ODDS_TAPE_FILE)

    signals_only = [s for s in signals if s.get("type") == "signal"]
    print(f"Loaded {len(signals_only)} signals and {len(tape)} tape entries")

    tape_index = build_tape_index(tape)
    print(f"Tape covers {len(tape_index)} markets")
    print()

    analyse(signals_only, tape_index, args)


if __name__ == "__main__":
    main()
