#!/usr/bin/env python3
"""
BLOCK 3 - MARKET CONTEXT FILTER
================================
Python port of the filtering logic in the RealTest script (intrada.rts).

Reads:
  - Raw trade CSVs from the Python backtester (e.g. SPY_trades.csv, QQQ_trades.csv)
  - 1-minute market data CSVs (e.g. market-data/SPY.csv) to build daily bars
  - Optional *_events.csv files (Type 300 = ATR) to reuse the EXACT ATR values
    the backtester used (recommended, mirrors the .rts note about matching config)

Applies:
  - BRK_Context filter from the .rts file:
        BRK_Context: abs(NextOpen/C) > HHV(abs(O/C[1]), 5)
    i.e. keep a trade only if its day's opening gap ratio (Open_D / Close_{D-1})
    is greater than the highest such ratio over the previous GAP_LOOKBACK days.
    (Comment in .rts: "trade only when there's the highest positive opening gap
    over last 5 days".)

Writes:
  - filtered_trades.csv   : same columns as input trade files, filtered rows only
  - daily_context.csv     : Symbol, Date, Open, Close, ATR, GapRatio, ContextOK
                            (consumed by build_portfolio.py so it never needs
                            minute data)

Usage:
  python3 filter_trades.py --data-dir market-data --trades-dir . --out-dir rt_python
  python3 filter_trades.py --no-filter        # passthrough (keep all trades)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIG - IMPORTANT: must match your Python backtester CONFIG (see .rts notes)
# ----------------------------------------------------------------------------
ATR_PERIOD = 5          # .rts: AtrPeriod: 5
GAP_LOOKBACK = 5        # .rts: HHV(abs(O/C[1]), 5)
SESSION_TZ = "America/New_York"
SESSION_START = "09:30"
SESSION_END = "16:00"
EVENTS_ATR_TYPE = 300   # Type code for ATR rows in *_events.csv


# ----------------------------------------------------------------------------
# Data loading helpers
# ----------------------------------------------------------------------------
COLUMN_ALIASES = {
    "o": "Open", "open": "Open",
    "h": "High", "high": "High",
    "l": "Low", "low": "Low",
    "c": "Close", "close": "Close",
    "v": "Volume", "volume": "Volume",
    "t": "timestamp", "time": "timestamp", "timestamp": "timestamp",
    "date": "timestamp", "datetime": "timestamp",
}


def load_minute_csv(path: str) -> pd.DataFrame:
    """Load a 1-minute bar CSV (Alpaca style or generic OHLCV) into a
    tz-aware DataFrame indexed by timestamp."""
    df = pd.read_csv(path)
    df.columns = [COLUMN_ALIASES.get(c.strip().lower(), c) for c in df.columns]
    if "timestamp" not in df.columns:
        # maybe the timestamp is the (unnamed) first column / index
        df = pd.read_csv(path, index_col=0)
        df.columns = [COLUMN_ALIASES.get(c.strip().lower(), c) for c in df.columns]
        df.index.name = "timestamp"
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df


def minute_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into daily bars of the regular US session,
    in exchange time (America/New_York), so each bucket = one trading day."""
    local = minute_df.tz_convert(SESSION_TZ)
    local = local.between_time(SESSION_START, SESSION_END)
    daily = pd.DataFrame({
        "Open": local["Open"].resample("1D").first(),
        "High": local["High"].resample("1D").max(),
        "Low": local["Low"].resample("1D").min(),
        "Close": local["Close"].resample("1D").last(),
    }).dropna()
    daily.index = daily.index.date  # plain dates as keys
    daily.index.name = "Date"
    return daily


def compute_atr(daily: pd.DataFrame, period: int) -> pd.Series:
    """Classic ATR (Wilder smoothing) on daily bars. Only used as a FALLBACK
    when no events file provides the backtester's own ATR values."""
    prev_close = daily["Close"].shift(1)
    tr = pd.concat([
        daily["High"] - daily["Low"],
        (daily["High"] - prev_close).abs(),
        (daily["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder's smoothing = EMA with alpha 1/period
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def load_events_atr(trades_dir: str) -> pd.DataFrame:
    """Load ATR values exported by the backtester (*_events.csv, Type 300).
    Returns DataFrame with columns Symbol, Date, ATR (may be empty)."""
    frames = []
    for path in glob.glob(os.path.join(trades_dir, "*_events.csv")):
        ev = pd.read_csv(path)
        ev.columns = [c.strip() for c in ev.columns]
        ev = ev[ev["Type"] == EVENTS_ATR_TYPE].copy()
        ev["Date"] = pd.to_datetime(ev["Date"]).dt.date
        frames.append(ev[["Symbol", "Date", "Value"]].rename(columns={"Value": "ATR"}))
    if not frames:
        return pd.DataFrame(columns=["Symbol", "Date", "ATR"])
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["Symbol", "Date"])
    return out


# ----------------------------------------------------------------------------
# The context filter (port of BRK_Context)
# ----------------------------------------------------------------------------
def build_context(daily: pd.DataFrame) -> pd.DataFrame:
    """For each day D compute:
         GapRatio  = Open_D / Close_{D-1}            (abs(NextOpen/C) in .rts)
         ContextOK = GapRatio > max(GapRatio_{D-1} ... GapRatio_{D-lookback})
       which is the .rts condition  abs(NextOpen/C) > HHV(abs(O/C[1]), 5)
       evaluated from the perspective of the trade day."""
    gap = (daily["Open"] / daily["Close"].shift(1)).abs()
    prior_max = gap.shift(1).rolling(GAP_LOOKBACK).max()
    ctx = pd.DataFrame({
        "Open": daily["Open"],
        "Close": daily["Close"],
        "GapRatio": gap,
        "ContextOK": gap > prior_max,
    })
    ctx["ContextOK"] = ctx["ContextOK"].fillna(False)
    return ctx


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Block 3: filter raw breakout trades by market context")
    ap.add_argument("--data-dir", default="market-data", help="directory with 1-minute CSVs (SPY.csv, ...)")
    ap.add_argument("--trades-dir", default="RT_trades", help="directory with trade CSV(s) and optional *_events.csv")
    ap.add_argument("--trades-file", default=None,
                    help="single combined trades CSV (e.g. BRK_trades.csv). "
                         "If given, ONLY this file is read. If omitted, all *_trades.csv "
                         "in --trades-dir are pooled (per-symbol files).")
    ap.add_argument("--out-dir", default="RT_filtered", help="output directory")
    ap.add_argument("--no-filter", action="store_true", help="keep all trades (baseline / unfiltered run)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load raw trades -----------------------------------------------
    # Two supported layouts:
    #   (a) one combined file (BRK_trades.csv) holding every symbol  -> --trades-file
    #   (b) one file per symbol (SPY_trades.csv, QQQ_trades.csv, ...) -> default glob
    # Do NOT mix both in the same folder under the glob, or trades double-count.
    if args.trades_file:
        path = args.trades_file
        if not os.path.isabs(path):
            path = os.path.join(args.trades_dir, path)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
        trades = pd.read_csv(path)
    else:
        trade_files = sorted(glob.glob(os.path.join(args.trades_dir, "*_trades.csv")))
        if not trade_files:
            print(f"ERROR: no *_trades.csv found in {args.trades_dir}", file=sys.stderr)
            return 1
        print(f"Pooling {len(trade_files)} trade file(s): {[os.path.basename(f) for f in trade_files]}")
        trades = pd.concat([pd.read_csv(f) for f in trade_files], ignore_index=True)

    # safety net: drop exact-duplicate trades (guards against a combined file
    # AND its per-symbol components both being present)
    key = ["Symbol", "Side", "DateIn", "PriceIn", "DateOut", "PriceOut"]
    before = len(trades)
    trades = trades.drop_duplicates(subset=[c for c in key if c in trades.columns])
    if len(trades) < before:
        print(f"WARNING: dropped {before - len(trades)} duplicate trade rows "
              "(combined + per-symbol files both present?)", file=sys.stderr)

    trades["DateIn"] = pd.to_datetime(trades["DateIn"])
    trades["DateOut"] = pd.to_datetime(trades["DateOut"])
    trades["TradeDate"] = trades["DateIn"].dt.date
    symbols = sorted(trades["Symbol"].unique())
    print(f"Loaded {len(trades)} raw trades for {symbols}")

    # ---- backtester ATR (preferred source), fallback: compute ----------
    events_atr = load_events_atr(args.trades_dir)
    if not events_atr.empty:
        print(f"Using {len(events_atr)} ATR values from *_events.csv (Type {EVENTS_ATR_TYPE})")

    # ---- daily bars + context per symbol --------------------------------
    ctx_frames = []
    for sym in symbols:
        path = os.path.join(args.data_dir, f"{sym}.csv")
        if not os.path.exists(path):
            print(f"WARNING: no minute data for {sym} at {path}; its trades will be dropped", file=sys.stderr)
            continue
        daily = minute_to_daily(load_minute_csv(path))
        ctx = build_context(daily)
        ctx["Symbol"] = sym

        # attach ATR: events file first, computed fallback
        ev = events_atr[events_atr["Symbol"] == sym].set_index("Date")["ATR"] if not events_atr.empty else pd.Series(dtype=float)
        computed = compute_atr(daily, ATR_PERIOD)
        ctx["ATR"] = [ev.get(d, computed.get(d, np.nan)) for d in ctx.index]

        ctx_frames.append(ctx.reset_index().rename(columns={"index": "Date"}))

    if not ctx_frames:
        print("ERROR: no daily context could be built", file=sys.stderr)
        return 1
    context = pd.concat(ctx_frames, ignore_index=True)
    context = context[["Symbol", "Date", "Open", "Close", "ATR", "GapRatio", "ContextOK"]]
    ctx_path = os.path.join(args.out_dir, "daily_context.csv")
    context.to_csv(ctx_path, index=False)
    print(f"Wrote {ctx_path} ({len(context)} symbol-days)")

    # ---- apply filter to trades -----------------------------------------
    merged = trades.merge(
        context[["Symbol", "Date", "ContextOK"]],
        left_on=["Symbol", "TradeDate"], right_on=["Symbol", "Date"],
        how="left",
    )
    merged["ContextOK"] = merged["ContextOK"].fillna(False)

    if args.no_filter:
        kept = merged
        print("Filter DISABLED (--no-filter): keeping all trades")
    else:
        kept = merged[merged["ContextOK"]]

    out_cols = ["Symbol", "Strategy", "Side", "DateIn", "QtyIn", "PriceIn",
                "DateOut", "QtyOut", "PriceOut", "FeesIn", "FeesOut"]
    out_path = os.path.join(args.out_dir, "filtered_trades.csv")
    kept[out_cols].to_csv(out_path, index=False)

    n_raw, n_kept = len(trades), len(kept)
    pct = 100.0 * n_kept / n_raw if n_raw else 0
    print(f"Wrote {out_path}: kept {n_kept}/{n_raw} trades ({pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())