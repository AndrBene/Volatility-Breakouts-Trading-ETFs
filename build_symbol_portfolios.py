#!/usr/bin/env python3
"""
BLOCK 4 (per-symbol) - INDEPENDENT PORTFOLIO PER ETF
=====================================================
Runs a SEPARATE, self-contained portfolio simulation for each selected ETF.

Unlike build_portfolio.py (one shared account across all symbols), this script
gives EACH symbol its own full account of ACCOUNT_SIZE and simulates it in
isolation, as if you traded that ETF alone. There is no interaction between
symbols: no shared capital, no shared profits, no cross-symbol sizing.

Money management per symbol (RealTest BRK_moneymanagement, per-symbol scope):
    risk_dollars  = equity * RISK_PER_TRADE / 100          # equity of THIS symbol
    stop_distance = STOPLOSS_MULTIPLIER * ATR(trade day)
    shares        = floor(risk_dollars / stop_distance)    # capped at MAX_POSITION_PCT
    commission    = max(0.005 * shares, 1)   charged on entry AND exit
Equity compounds within the symbol: each closed trade's PnL feeds the next
trade's sizing.

Reads:
  - <in-dir>/filtered_trades.csv   (from filter_trades.py)
  - <in-dir>/daily_context.csv     (from filter_trades.py; ATR per symbol/day)

Writes, for each selected symbol SYM:
  - <out-dir>/SYM_portfolio.csv    one row per executed trade, with Shares,
                                   Commission, PnL, EquityAfter
And a small summary:
  - <out-dir>/symbol_portfolios_summary.csv

Config:
  SYMBOLS  = None  -> simulate every symbol present in filtered_trades.csv
           = ["SPY", "QQQ"]  -> only these
  (also overridable on the CLI with --symbols SPY QQQ)

Usage:
  python3 build_symbol_portfolios.py --in-dir RT_filtered --out-dir RT_output
  python3 build_symbol_portfolios.py --symbols SPY QQQ
"""

import argparse
import math
import os
import sys

import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG - IMPORTANT: keep in sync with filter_trades.py / your backtester
# ----------------------------------------------------------------------------
ACCOUNT_SIZE = 25_000.0        # full account given to EACH symbol independently
RISK_PER_TRADE = 1.5           # % of that symbol's equity risked per trade
STOPLOSS_MULTIPLIER = 0.33     # stop distance = this * ATR
COMMISSION_PER_SHARE = 0.005   # max(0.005 * shares, 1) per side
COMMISSION_MIN = 1.0
MAX_POSITION_PCT = 100.0       # cap one position's value at this % of equity

# Which symbols to simulate. None = all symbols found in filtered_trades.csv.
# Override on the command line with --symbols SPY QQQ IWM
SYMBOLS: list[str] | None = None


def commission(shares: int) -> float:
    return max(COMMISSION_PER_SHARE * shares, COMMISSION_MIN)


def simulate_symbol(sym_trades: pd.DataFrame, atr_lookup: dict) -> tuple[pd.DataFrame, dict]:
    """Run an independent, compounding simulation for a single symbol.

    sym_trades: filtered trades for ONE symbol (any order; sorted here).
    atr_lookup: {(Symbol, date): ATR}
    Returns (per-trade DataFrame, summary dict).
    """
    sym_trades = sym_trades.sort_values("DateIn").reset_index(drop=True)
    equity = ACCOUNT_SIZE
    pending: list[tuple[pd.Timestamp, float]] = []  # (DateOut, net pnl)
    rows = []
    skipped_no_atr = 0

    for t in sym_trades.itertuples():
        # realize exits that completed before this entry (compounding)
        pending.sort(key=lambda x: x[0])
        while pending and pending[0][0] <= t.DateIn:
            equity += pending.pop(0)[1]

        atr = atr_lookup.get((t.Symbol, t.DateIn.date()))
        if atr is None or not (atr == atr) or atr <= 0:  # missing or NaN
            skipped_no_atr += 1
            continue

        # ---- position sizing (per-symbol equity) ----------------------
        risk_dollars = equity * RISK_PER_TRADE / 100.0 # willinng loss per trade (N shares)
        stop_distance = STOPLOSS_MULTIPLIER * atr # willing loss per share
        shares = math.floor(risk_dollars / stop_distance)

        max_shares = math.floor(equity * MAX_POSITION_PCT / 100.0 / t.PriceIn)
        shares = min(shares, max_shares)
        if shares < 1:
            continue

        side_sign = 1 if str(t.Side).lower() == "long" else -1
        fees = commission(shares) * 2  # entry + exit
        gross = side_sign * (t.PriceOut - t.PriceIn) * shares
        net = gross - fees

        pending.append((t.DateOut, net))
        rows.append({
            "Symbol": t.Symbol, "Strategy": t.Strategy, "Side": t.Side,
            "DateIn": t.DateIn, "PriceIn": t.PriceIn,
            "DateOut": t.DateOut, "PriceOut": t.PriceOut,
            "ATR": atr, "Shares": shares,
            "RiskDollars": round(risk_dollars, 2),
            "Commission": round(fees, 2),
            "GrossPnL": round(gross, 2), "PnL": round(net, 2),
            "EquityAtEntry": round(equity, 2),
        })

    # flush remaining open trades
    for _, pnl in sorted(pending):
        equity += pnl

    if not rows:
        return pd.DataFrame(), {
            "Symbol": sym_trades["Symbol"].iloc[0] if len(sym_trades) else "?",
            "Trades": 0, "FinalEquity": ACCOUNT_SIZE, "NetProfit": 0.0,
            "ReturnPct": 0.0, "SkippedNoATR": skipped_no_atr,
        }

    df = pd.DataFrame(rows)
    # running equity after each exit (chronological by exit)
    by_exit = df.sort_values("DateOut").copy()
    by_exit["EquityAfter"] = ACCOUNT_SIZE + by_exit["PnL"].cumsum()
    df = by_exit.sort_values("DateIn").reset_index(drop=True)

    net = df["PnL"].sum()
    summary = {
        "Symbol": df["Symbol"].iloc[0],
        "Trades": len(df),
        "FinalEquity": round(ACCOUNT_SIZE + net, 2),
        "NetProfit": round(net, 2),
        "ReturnPct": round(100.0 * net / ACCOUNT_SIZE, 2),
        "SkippedNoATR": skipped_no_atr,
    }
    return df, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Independent portfolio simulation per ETF")
    ap.add_argument("--in-dir", default="RT_filtered",
                    help="directory with filtered_trades.csv + daily_context.csv")
    ap.add_argument("--out-dir", default="RT_output", help="output directory (default: --in-dir)")
    ap.add_argument("--trades-file", default="filtered_trades.csv",
                    help="trade list inside --in-dir")
    ap.add_argument("--symbols", nargs="*", default=None,
                    help="symbols to simulate (default: SYMBOLS config, else all present)")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.in_dir
    os.makedirs(args.out_dir, exist_ok=True)

    trades = pd.read_csv(os.path.join(args.in_dir, args.trades_file),
                         parse_dates=["DateIn", "DateOut"])
    if trades.empty:
        print("No trades to process.", file=sys.stderr)
        return 1

    context = pd.read_csv(os.path.join(args.in_dir, "daily_context.csv"),
                          parse_dates=["Date"])
    atr_lookup = {(r.Symbol, r.Date.date()): r.ATR for r in context.itertuples()}

    # resolve symbol selection: CLI > config > all present
    present = sorted(trades["Symbol"].unique())
    wanted = args.symbols if args.symbols is not None else SYMBOLS
    if wanted is None:
        selected = present
    else:
        selected = [s for s in wanted if s in present]
        missing = [s for s in wanted if s not in present]
        if missing:
            print(f"WARNING: requested symbols not in trades: {missing}", file=sys.stderr)
    if not selected:
        print(f"ERROR: no matching symbols. Present: {present}", file=sys.stderr)
        return 1

    print(f"Simulating {len(selected)} symbol(s), each with ${ACCOUNT_SIZE:,.0f}: {selected}\n")

    summaries = []
    for sym in selected:
        sym_trades = trades[trades["Symbol"] == sym]
        df, summary = simulate_symbol(sym_trades, atr_lookup)
        summaries.append(summary)

        out_path = os.path.join(args.out_dir, f"{sym}_portfolio.csv")
        df.to_csv(out_path, index=False)

        print(f"  {sym:<6} {summary['Trades']:>4d} trades  "
              f"${ACCOUNT_SIZE:,.0f} -> ${summary['FinalEquity']:>12,.2f}  "
              f"({summary['ReturnPct']:+.2f}%)  -> {os.path.basename(out_path)}"
              + (f"   [{summary['SkippedNoATR']} skipped: no ATR]"
                 if summary["SkippedNoATR"] else ""))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "symbol_portfolios_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())