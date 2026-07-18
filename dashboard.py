#!/usr/bin/env python3
"""
BLOCK 5 (interactive) - STREAMLIT DASHBOARD (per-symbol portfolios)
====================================================================
Reads the per-ETF portfolios from build_symbol_portfolios.py:

    <run-dir>/SYM_portfolio.csv               one file per symbol

Each symbol is an INDEPENDENT portfolio that started at the full ACCOUNT_SIZE.
Pick one ETF to analyse; optionally overlay other ETFs and/or a buy-and-hold
benchmark (daily closes read from the market-data folder).

Run:
    pip install streamlit plotly scipy
    streamlit run dashboard.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# ACCOUNT_SIZE = 25_000.0        # must match build_symbol_portfolios.py
TRADING_DAYS_PER_YEAR = 252

st.set_page_config(page_title="Intraday Breakout - Per-ETF Portfolios", layout="wide")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data
def list_symbol_files(run_dir: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*_portfolio.csv"))):
        sym = os.path.basename(path).replace("_portfolio.csv", "")
        out[sym] = path
    return out


@st.cache_data
def load_symbol(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["DateIn", "DateOut"])


@st.cache_data
def load_daily_close(data_dir: str, symbol: str) -> pd.Series | None:
    """Daily close series for buy-and-hold, from market-data/<SYM>.csv (1-min bars)."""
    path = os.path.join(data_dir, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    tcol = cols.get("timestamp") or cols.get("time") or cols.get("date") or df.columns[0]
    ccol = cols.get("c") or cols.get("close")
    if ccol is None:
        return None
    ts = pd.to_datetime(df[tcol], utc=True)
    close = pd.Series(df[ccol].values, index=ts).sort_index()
    daily = close.resample("1D").last().dropna()
    daily.index = pd.to_datetime(daily.index.date)
    return daily


def buy_hold_curve(daily_close: pd.Series,
                   start: float) -> pd.Series:
    """Equity from buying `start` worth at the first close and holding."""
    shares = start / daily_close.iloc[0]
    return (shares * daily_close).rename("Buy & hold")


def equity_from_trades(trades: pd.DataFrame, pnl_col: str,
                       start: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    daily = trades.sort_values("DateOut").groupby(trades["DateOut"].dt.date)[pnl_col].sum()
    curve = start + daily.cumsum()
    curve.index = pd.to_datetime(curve.index)
    return curve


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def max_drawdown(equity: pd.Series):
    peak = equity.cummax()
    dd = equity - peak
    return dd.min(), (dd / peak).min()


def curve_stats(curve: pd.Series, account_size: int) -> dict:
    """Performance stats from an equity curve alone (works for buy-and-hold too)."""
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    final = curve.iloc[-1]
    net = final - account_size
    cagr = (final / account_size) ** (1 / years) - 1 if final > 0 else np.nan
    dd_abs, dd_pct = max_drawdown(curve)
    daily_ret = curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_ret.std() > 0 else np.nan)
    mar = (cagr / abs(dd_pct)) if (dd_pct and pd.notna(cagr)) else np.nan
    return {"years": years, "final": final, "net": net, "total_return": net / account_size,
            "cagr": cagr, "dd_abs": dd_abs, "dd_pct": dd_pct, "sharpe": sharpe, "mar": mar}


def trade_stats(trades: pd.DataFrame, pnl_col: str) -> dict:
    pnl = trades[pnl_col]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    longs = (trades["Side"].str.lower() == "long").sum()
    shorts = (trades["Side"].str.lower() == "short").sum()
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.inf
    n_top = max(1, len(pnl) // 20)
    net = pnl.sum()
    return {
        "n": len(pnl), "longs": int(longs), "shorts": int(shorts),
        "win_rate": len(wins) / len(pnl) if len(pnl) else np.nan,
        "expectancy": pnl.mean(), "median": pnl.median(), "std": pnl.std(),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "best": pnl.max(), "worst": pnl.min(),
        "pf": pf, "commissions": trades["Commission"].sum(),
        "skew": stats.skew(pnl), "kurt": stats.kurtosis(pnl),
        "top5_share": pnl.nlargest(n_top).sum() / net if net > 0 else np.nan,
        "exp_wo_best": pnl.drop(pnl.idxmax()).mean() if len(pnl) > 1 else np.nan,
    }

def accounting_fmt(x):
    if pd.isna(x):
        return ""
    return f"({abs(x):,.2f}$)" if x < 0 else f"{x:,.2f}$"

def red_if_negative(x):
    return "color: red" if (pd.notna(x) and x < 0) else ""

def color_side(x):
    if x == "Long":
        return "color: red"
    if x == "Short":
        return "color: green"
    return ""

# ----------------------------------------f-----------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Run")
run_dir = st.sidebar.text_input("Run directory", "RT_output")
data_dir = st.sidebar.text_input("Market-data directory (for buy & hold)", "market-data")

files = list_symbol_files(run_dir)
if not files:
    st.error(f"No `*_portfolio.csv` files found in `{run_dir}`. "
             "Run build_symbol_portfolios.py first.")
    st.stop()

choice = st.sidebar.selectbox("ETF", list(files.keys()))
pnl_col = st.sidebar.radio("P&L column", ["PnL", "GrossPnL"],
                           help="PnL is net of commissions; GrossPnL is before costs.")

trades = load_symbol(files[choice])
st.sidebar.header("Slice")
sides = st.sidebar.multiselect("Side", sorted(trades.Side.unique()),
                               default=sorted(trades.Side.unique()))
view = trades[trades.Side.isin(sides)]
if view.empty:
    st.warning("No trades match the current slice.")
    st.stop()

curve = equity_from_trades(view, pnl_col, trades["EquityAtEntry"].iloc[0])
cs = curve_stats(curve, trades["EquityAtEntry"].iloc[0])
ts = trade_stats(view, pnl_col)

# ---------------------------------------------------------------------------
# Header + full PORTFOLIO ANALYSIS block
# ---------------------------------------------------------------------------
st.header("Portfolio analysis")
st.caption(f"Period {curve.index[0].date()} -> {curve.index[-1].date()} "
           f"({cs['years']:.2f} years)")

st.subheader("Performance")
p = st.columns(4)
p[0].metric("Starting equity", f"${trades["EquityAtEntry"].iloc[0]:,.2f}")
p[1].metric("Net profit", f"${cs['net']:,.2f}")
p[2].metric("Final equity", f"${cs['final']:,.2f}")
p[3].metric("Total return", f"{cs['total_return']:.1%}")
p = st.columns(4)
p[0].metric("CAGR", f"{cs['cagr']:.1%}" if pd.notna(cs["cagr"]) else "-")
p[1].metric("Max drawdown ($)", f"{cs['dd_pct']:.1%}", f"{cs['dd_abs']:,.2f}")
p[2].metric("MAR (CAGR/|MaxDD%|)", f"{cs['mar']:.2f}" if pd.notna(cs["mar"]) else "-")
p[3].metric("Sharpe (daily, ann.)", f"{cs['sharpe']:.2f}" if pd.notna(cs["sharpe"]) else "-")

st.subheader("Trades")
q = st.columns(4)
q[0].metric("Number of trades", f"{ts['n']}")
q[1].metric("Long / Short", f"{ts['longs']} / {ts['shorts']}")
q[2].metric("Win rate", f"{ts['win_rate']:.1%}")
q[3].metric("Average trade (exp.)", f"${ts['expectancy']:,.2f}")
q = st.columns(4)
q[0].metric("Average win", f"${ts['avg_win']:,.2f}")
q[1].metric("Average loss", f"${ts['avg_loss']:,.2f}")
q[2].metric("Best / worst", f"\\${ts['best']:,.2f} / \\${ts['worst']:,.2f}")
q[3].metric("Profit factor", f"{ts['pf']:.2f}")
q = st.columns(4)
q[0].metric("Total commissions", f"${ts['commissions']:,.2f}")
q[1].metric("Skew", f"{ts['skew']:.2f}")
q[2].metric("Excess kurtosis", f"{ts['kurt']:.2f}")
q[3].metric("Median trade", f"${ts['median']:,.2f}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_eq, tab_dist, tab_frag, tab_slice, tab_tbl = st.tabs(
    ["Equity curve", "P&L distribution", "Fragility", "By period", "Trades"])

with tab_eq:
    other_syms = [s for s in files if s != choice]
    col1, col2, _ = st.columns([1, 1, 2])
    show_bh = col1.checkbox("Buy & hold", value=False)
    compare_syms = col2.multiselect(
        "Compare with",
        options=sorted(other_syms),
        default=[],
        key=f"compare_multiselect_{choice}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.06,
                        subplot_titles=(f"Equity - {choice}", "Drawdown"))

    # fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=f"{choice} strategy",
    #                          mode="lines", line=dict(width=2.6),
    #                          hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)
       # --- selected ETF strategy: colour by side selection ------------------
    long_sel = "Long" in sides
    short_sel = "Short" in sides
    BLUE = "#1f77b4"

    if long_sel and short_sel:
        # both sides -> three lines: green long, red short, blue combined
        long_curve = equity_from_trades(view[view.Side == "Long"], pnl_col, trades["EquityAtEntry"].iloc[0])
        short_curve = equity_from_trades(view[view.Side == "Short"], pnl_col, trades["EquityAtEntry"].iloc[0])
        if not long_curve.empty:
            fig.add_trace(go.Scatter(x=long_curve.index, y=long_curve.values,
                                     name=f"{choice} (Long)", mode="lines",
                                     line=dict(width=1.0, color="green"),
                                     hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)
        if not short_curve.empty:
            fig.add_trace(go.Scatter(x=short_curve.index, y=short_curve.values,
                                     name=f"{choice} (Short)", mode="lines",
                                     line=dict(width=1.0, color="red"),
                                     hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values,
                                 name=f"{choice} (Combined)", mode="lines",
                                 line=dict(width=2.6, color=BLUE),
                                 hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)
    else:
        # single side -> one blue line, labelled with the side
        side_label = "Long" if long_sel else "Short"
        fig.add_trace(go.Scatter(x=curve.index, y=curve.values,
                                 name=f"{choice} ({side_label})", mode="lines",
                                 line=dict(width=2.6, color=BLUE),
                                 hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)


    if show_bh:
        dclose = load_daily_close(data_dir, choice)
        if dclose is None:
            st.warning(f"No price data for {choice} in `{data_dir}` - "
                       "buy & hold unavailable.")
        else:
            bh = buy_hold_curve(dclose, trades["EquityAtEntry"].iloc[0])
            bh = bh[(bh.index >= curve.index[0]) & (bh.index <= curve.index[-1])]
            fig.add_trace(go.Scatter(x=bh.index, y=bh.values,
                                     name=f"{choice} buy & hold", mode="lines",
                                     line=dict(width=1.8, dash="dash"),
                                     hovertemplate="$%{y:,.0f}<extra></extra>"),
                          row=1, col=1)

    for sym in compare_syms:
        d = load_symbol(files[sym])
        d = d[d.Side.isin(sides)]
        if d.empty:
            continue
        c_ = equity_from_trades(d, pnl_col, d["EquityAtEntry"].iloc[0])
        sym_cs = curve_stats(c_, d["EquityAtEntry"].iloc[0])
        fig.add_trace(go.Scatter(x=c_.index, y=c_.values,
                                 name=f"{sym} (sharpe: {sym_cs['sharpe']:,.2f})",
                                 mode="lines", line=dict(width=1.4),
                                 hovertemplate="$%{y:,.0f}<extra></extra>"), row=1, col=1)

    fig.add_hline(y=trades["EquityAtEntry"].iloc[0], line_dash="dot", line_width=1, opacity=0.4, row=1, col=1)

    peak = curve.cummax()
    fig.add_trace(go.Scatter(x=curve.index, y=(curve - peak) / peak * 100,
                             name="Drawdown %", fill="tozeroy",
                             showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width='stretch')

    if show_bh:
        dclose = load_daily_close(data_dir, choice)
        if dclose is not None:
            bh = buy_hold_curve(dclose, trades["EquityAtEntry"].iloc[0])
            bh = bh[(bh.index >= curve.index[0]) & (bh.index <= curve.index[-1])]
            bhs = curve_stats(bh, trades["EquityAtEntry"].iloc[0])
            cmp_df = pd.DataFrame({
                f"{choice} strategy": [cs["net"], cs["total_return"], cs["cagr"],
                                       cs["dd_pct"], cs["sharpe"], cs["mar"]],
                f"{choice} buy & hold": [bhs["net"], bhs["total_return"], bhs["cagr"],
                                         bhs["dd_pct"], bhs["sharpe"], bhs["mar"]],
            }, index=["Net profit $", "Total return", "CAGR",
                      "Max drawdown %", "Sharpe", "MAR"])
            st.dataframe(cmp_df.style.format({
                f"{choice} strategy": lambda v: f"{v:,.2f}",
                f"{choice} buy & hold": lambda v: f"{v:,.2f}"}),
                width='stretch')
            st.caption(
                "Buy & hold is price-only (no dividends), so it understates the benchmark "
                "slightly. Compare on risk too: an intraday strategy that is flat overnight "
                "should be judged on drawdown and Sharpe, not raw return alone."
            )

with tab_dist:
    pnl_s = view[pnl_col]
    pnl_a = pnl_s.to_numpy()
    # lo, hi = \
    #     pnl_s.mean() - 1.96 * pnl_s.std() / np.sqrt(len(pnl_s)),  \
    #     pnl_s.mean() + 1.96 * pnl_s.std() / np.sqrt(len(pnl_s))  \
    # lo, hi = stats.t.interval(0.95, len(pnl_s)-1, loc=np.mean(pnl_s), scale=stats.sem(pnl_s))
    se = pnl_s.std(ddof=1) / np.sqrt(len(pnl_s))
    lo, hi = pnl_s.mean() - 1.96*se, pnl_s.mean() + 1.96*se

    # ---- summary of the trade distribution ------------------------------
    m = st.columns(4)
    m[0].metric("Mean (expectancy)", f"${pnl_s.mean():,.2f}")
    m[1].metric("Median", f"${pnl_s.median():,.2f}")
    m[2].metric("Std dev", f"${pnl_s.std():,.2f}")
    m[3].metric("95% Confidence Interval", f"[\\${lo:.2f}, \\${hi:.2f}]")

    # ---- histogram of individual trades ---------------------------------
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pnl_s, nbinsx=60, name="Trades",
                               histnorm="probability density"))
    xs = np.linspace(pnl_s.min(), pnl_s.max(), 300)
    # fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, pnl_s.mean(), pnl_s.std()),
    #                          name="Normal fit", mode="lines"))
    fig.add_vline(x=pnl_s.mean(), line_dash="dash", line_color="red",
                  annotation_text=f"mean ${pnl_s.mean():.2f}")
    fig.add_vline(x=pnl_s.median(), line_dash="dot", line_color="green",
                  annotation_text=f"median ${pnl_s.median():.2f}")
    fig.add_vline(x=0, line_color="black", opacity=0.4)
    fig.update_layout(height=460, xaxis_title=f"{pnl_col} per trade ($)",
                      yaxis_title="Density", bargap=0.02)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"Skew {ts['skew']:.2f}, excess kurtosis {ts['kurt']:.2f}. A stop-loss breakout "
        "system should show positive skew (losses truncated, winners run). Mean above "
        "median is the expected signature - most trades lose a little, a few win big."
    )

with tab_frag:
    pnl = view[pnl_col].sort_values(ascending=False).reset_index(drop=True)
    rows = []
    for k in [0, 1, 3, 5, 10]:
        if k < len(pnl):
            rows.append({"Trades dropped": k, "Expectancy": pnl.iloc[k:].mean(),
                         "Net profit": pnl.iloc[k:].sum()})
    drop_df = pd.DataFrame(rows)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = go.Figure(go.Bar(x=drop_df["Trades dropped"].astype(str),
                               y=drop_df["Expectancy"]))
        fig.add_hline(y=0, line_color="red")
        fig.update_layout(height=380, xaxis_title="Best trades removed",
                          yaxis_title="Expectancy ($)")
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.metric("Top 5% of trades ->", f"{ts['top5_share']:.0%} of net profit"
                  if pd.notna(ts["top5_share"]) else "-")
        st.metric("Expectancy w/o best trade", f"${ts['exp_wo_best']:.2f}")
        st.metric("Total commissions", f"${ts['commissions']:,.0f}")
    st.caption(
        "If removing a handful of trades kills the edge, the backtest measured a few "
        "lucky days rather than a market regularity. High kurtosis predicts this."
    )

with tab_slice:
    by_year = view.groupby(view.DateIn.dt.year)[pnl_col].agg(["count", "sum", "mean"])
    by_year.columns = ["Trades", "Net profit", "Expectancy"]
    by_side = view.groupby("Side")[pnl_col].agg(["count", "sum", "mean"])
    by_side.columns = ["Trades", "Net profit", "Expectancy"]
    c1, c2 = st.columns(2)
    c1.subheader("By year")
    c1.dataframe(by_year.style.format("{:,.2f}"), width='stretch')
    c2.subheader("By side")
    c2.dataframe(by_side.style.format("{:,.2f}"), width='stretch')
    st.caption("An edge concentrated in one year or side is a warning sign.")

with tab_tbl:
    styled = (
        view.sort_values("DateIn").style
        .format(accounting_fmt, subset=["PnL"])
        .map(red_if_negative, subset=["PnL"])
        .map(color_side, subset=["Side"])
    )
    st.dataframe(styled, width='stretch', height=560)
    # st.dataframe(view.sort_values("DateIn"), width='stretch', height=560)
    st.download_button("Download as CSV", view.to_csv(index=False),
                       f"{choice}_trades.csv", "text/csv")