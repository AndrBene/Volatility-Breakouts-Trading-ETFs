# Volatility Breakout Trading: A Systematic Strategy for ETFs

This project backtests a systematic trading strategy on US index ETFs (SPY, QQQ, …), based on intraday volatility breakouts measured through ATR. The Python pipeline downloads historical 1-minute data from Alpaca APIs, simulates the breakout trades, sizes them into per-ETF portfolios, and analyzes the results in a Streamlit dashboard.

## The Strategy Rules

1. **Calculate Daily ATR**: Compute the Average True Range (ATR) with a period of 5 days.
2. **Determine Volatility Breakout Bands**:
   - **Long Entry**: Opening price + 0.33 \* ATR(5)
   - **Short Entry**: Opening price - 0.33 \* ATR(5)
3. **Stop-Loss**: For each trade, set a stop-loss at 0.33 \* ATR(5). Exit the trade if the market retraces to the opening price after breaching the breakout level.
4. **Trade Frequency**: In each market, execute only one long breakout attempt and one short breakout attempt per day.
5. **Risk Management**: Allocate a small portion of the account to each trade. The backtest uses 0.33% of the account per trade. Consequently, in a single market, you can enter up to two trades per day (one long and one short), risking a maximum of 0.66% of the account.
6. **Exit Strategy**: Exit either at the stop-loss or at the end of the trading day. The stop-loss remains fixed at the entry point (the opening price) throughout the day.

## Pipeline overview

```
 ┌─────────────────────────┐
 │ alpaca_download.py      │  1-minute bars from Alpaca  ─────► market-data/SPY.csv, QQQ.csv…
 └───────────┬─────────────┘
             ▼
 ┌─────────────────────────────────┐
 │ intraday_breakout_backtester.py │  apply rules ─► SPY_trades.csv (+ *_events.csv)
 └───────────┬─────────────────────┘
             ▼
 ┌─────────────────────────┐
 │ filter_trades.py        │  keep only good-context days ───► filtered_trades.csv
 │                         │                                   daily_context.csv
 └───────────┬─────────────┘
             ▼
 ┌───────────────────────────┐
 │ build_symbol_portfolios.py│  size positions  ────► SPY_portfolio.csv, QQQ_portfolio.csv
 └───────────┬───────────────┘                                  symbol_portfolios_summary.csv
             ▼
 ┌─────────────────────────┐
 │ dashboard.py            │  interactive analysis (Streamlit)
 └─────────────────────────┘
```

## Setup

```bash
# virtual environment
python -m venv venv
source venv/bin/activate

# dependencies
pip install -r requirements.txt
```

Alpaca API keys — set as environment variables before downloading:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

## Files

### 1. `alpaca_download.py` — data downloader

Downloads historical **1-minute bars** from the Alpaca Market Data API and stores them locally,
one CSV per ticker.

|             |                                                         |
| ----------- | ------------------------------------------------------- |
| **Reads**   | Alpaca API (SIP feed)                                   |
| **Writes**  | `market-data/SPY.csv`, `market-data/QQQ.csv`, …         |
| **Columns** | `timestamp, o, h, l, c, v` (UTC timestamps)             |
| **Config**  | timeframe (`1Min`), output dir, start date, ticker list |

**Why 1-minute data.** Daily bars tell you a day's high and low but not _which came first_ —
so they can't say whether a breakout level or the stop was touched first. Minute bars can.
The daily bars needed for ATR are derived from these by aggregation, so no separate download
is required.

---

### 2. `intraday_breakout_backtester.py` — trade generator

Replays the stored minute bars **bar by bar** for each ETF market separately, applies the breakout rules, and records every resulting trade.

|            |                                                     |
| ---------- | --------------------------------------------------- |
| **Reads**  | `market-data/<SYM>.csv`                             |
| **Writes** | `<SYM>_trades.csv`, `<SYM>_events.csv`              |
| **Config** | ATR period, stop multiplier, session times, tickers |

**What it does per day:** compute ATR from prior days → set the two trigger levels → walk the
day's minute bars → enter on a level cross → exit on stop or at end of day → write the trade.

**`<SYM>_trades.csv`** — one row per simulated trade:

```csv
Symbol,Strategy,Side,DateIn,QtyIn,PriceIn,DateOut,QtyOut,PriceOut,FeesIn,FeesOut
SPY,BRK,Long,2024-01-10 13:03:00,100,475.65,2024-01-10 16:00:00,100,476.62,0,0
```

Exits at exactly `16:00:00` are end-of-day exits; exits at other times are stop-outs. `Qty` is
a placeholder — real sizing happens later.

**`<SYM>_events.csv`** — daily indicator context, long format:

```csv
Symbol,Date,Type,Value
SPY,01/15/2024,300,4.1250
```

`Type 300 = ATR`. This exists because the ATR that _set_ each trade's levels isn't recoverable
from the trades file, but the later stages need it for sizing and filtering. Exporting it
guarantees every stage uses the **same** ATR values, rather than each recomputing its own
(and possibly differing) variant.

> ⚠️ **Critical:** this script's CONFIG (ATR period, stop multiplier) must match the constants
> in `filter_trades.py` and `build_symbol_portfolios.py`. They are not validated against each
> other — a mismatch fails silently.

---

### 3. `filter_trades.py` — market-context filter

Keeps only the trades taken on days whose **prior context** favours breakout follow-through.
This is where the edge is supposed to come from.

|            |                                                                                |
| ---------- | ------------------------------------------------------------------------------ |
| **Reads**  | `*_trades.csv` (or one combined file), `*_events.csv`, `market-data/<SYM>.csv` |
| **Writes** | `filtered_trades.csv`, `daily_context.csv`                                     |

**Config**

```python
ATR_PERIOD = 5          # AtrPeriod: 5
GAP_LOOKBACK = 5        # HHV(abs(O/C[1]), 5)
SESSION_TZ = "America/New_York"
SESSION_START, SESSION_END = "09:30", "16:00"
EVENTS_ATR_TYPE = 300   # ATR rows in *_events.csv
```

**The filter**:

```
abs(NextOpen / C) > HHV(abs(O / C[1]), 5)
```

In words: keep a trade only if its day's **opening gap ratio** exceeds the highest such ratio
of the previous 5 days. It's legitimate to use today's _open_ because the open is known before
any breakout level can be hit.

**What it builds along the way.** Minute bars → daily bars (converted to New York time,
regular session only, so each bucket is one real trading day) → gap ratio and rolling max.
ATR comes from `*_events.csv` when present (preferred — matches the backtester exactly), with
a computed Wilder ATR(5) as fallback.

**Two input layouts** — do not mix them under the glob, or every trade is counted twice:

```bash
# (a) one combined file containing all symbols
python filter_trades.py --trades-file BRK_trades.csv --trades-dir RT_trades --out-dir RT_output

# (b) one file per symbol (default: globs *_trades.csv)
python filter_trades.py --trades-dir . --out-dir RT_filtered

# baseline for comparison — keep every trade
python filter_trades.py --no-filter --out-dir rt_baseline
```

A dedup safety net drops exact-duplicate rows and warns if it finds any.

**Outputs**

- `filtered_trades.csv` — same columns as the input, surviving rows only.
- `daily_context.csv` — `Symbol, Date, Open, Close, ATR, GapRatio, ContextOK`. `build_symbol_portfolios.py` reads its ATR from here, so it never needs the minute data.

**Adding other filters.** The blueprint mentions three more, each one boolean column in
`build_context()`:

| Filter                 | Idea                                           | Sketch                                               |
| ---------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Narrow day (NR7)       | yesterday's range narrowest of last 7 → coiled | `rng.shift(1) == rng.shift(1).rolling(7).min()`      |
| Trend                  | trade with the prevailing trend                | `close.shift(1) > close.rolling(20).mean().shift(1)` |
| Volatility contraction | ATR below its own average → regime change due  | `atr5.shift(1) < atr5.shift(1).rolling(20).mean()`   |

Note the `.shift(1)` everywhere — each condition must be computable before the day starts.

---

### 4. `build_symbol_portfolios.py` — per-ETF portfolios

Runs a **separate, self-contained simulation for each ETF**. Every symbol gets its own full
account and compounds in isolation — no shared capital, no shared profits, no cross-symbol
sizing. Run SPY alone or alongside QQQ and SPY's numbers are identical.

|            |                                                        |
| ---------- | ------------------------------------------------------ |
| **Reads**  | `filtered_trades.csv`, `daily_context.csv`             |
| **Writes** | `<SYM>_portfolio.csv`, `symbol_portfolios_summary.csv` |

**Config**

```python
ACCOUNT_SIZE = 25_000.0        # full account given to EACH symbol independently
RISK_PER_TRADE = 1.5           # % of that symbol's equity risked per trade
STOPLOSS_MULTIPLIER = 0.33     # stop distance = this × ATR
COMMISSION_PER_SHARE = 0.005   # max(0.005 × shares, 1) per side
MAX_POSITION_PCT = 100.0       # cap one position's value at this % of equity
SYMBOLS = None                 # None = all symbols present; or ["SPY", "QQQ"]
```

**Position sizing**:

```python
risk_dollars  = equity * RISK_PER_TRADE / 100      # dollars willing to lose
stop_distance = STOPLOSS_MULTIPLIER * atr          # dollars lost per share if stopped
shares        = floor(risk_dollars / stop_distance)
shares        = min(shares, floor(equity * MAX_POSITION_PCT / 100 / price))
```

The logic: solve `shares × stop_distance = equity × 1.5%` so **every trade risks the same
fraction of the account**. Because the stop is ATR-relative, position size automatically
shrinks when volatility rises — dollar risk stays constant across regimes. Note price appears
only in the _cap_, not the risk formula: what a stop-out costs depends on the stop _distance_,
not the price level.

**Equity compounds within the symbol.** Trades are processed in entry order; a trade's P&L is
credited when it _exits_, so later trades are sized off earlier results.

**Usage**

```bash
python build_symbol_portfolios.py --in-dir RT_filtered --out-dir results --symbols SPY QQQ
```

**Output** — `<SYM>_portfolio.csv`, one row per executed trade:

`Symbol, Strategy, Side, DateIn, PriceIn, DateOut, PriceOut, ATR, Shares, RiskDollars,
Commission, GrossPnL, PnL, EquityAtEntry, EquityAfter`

where `PnL = GrossPnL − Commission` and `EquityAfter = ACCOUNT_SIZE + cumulative PnL`.

> ⚠️ **The `MAX_POSITION_PCT` cap probably binds most of the time.** On a \$25k account with
> \$475 SPY, the risk formula asks for ~275 shares (\$130k) but the cap allows ~52. When the cap
> binds you're risking far less than 1.5%, and the strategy is effectively
> _fixed-fraction-of-capital_, not risk-based. Measure it:
>
> ```python
> (port.Shares == port.EquityAtEntry // port.PriceIn).mean()   # % of trades capped
> ```
>
> Raise to `400.0` to model 4:1 intraday margin, or accept the capital-constrained variant.

---

### 5. `dashboard.py` — interactive analysis

Streamlit app over the per-symbol portfolios. Select one ETF; optionally overlay other ETFs
and a buy-and-hold benchmark.

```bash
pip install streamlit plotly scipy
streamlit run dashboard.py
```

|           |                                                                       |
| --------- | --------------------------------------------------------------------- |
| **Reads** | `<run-dir>/*_portfolio.csv`, `market-data/<SYM>.csv` (for buy & hold) |

**Header** — the full portfolio analysis: period, starting/net/final equity, total return,
CAGR, max drawdown (\$ and %), MAR, Sharpe; then trade count, long/short, win rate, expectancy,
average win/loss, best/worst, profit factor, commissions, skew, kurtosis, median.

**Tabs**

- **Equity curve** — strategy vs buy-and-hold (dashed) and any comparison ETFs, with a
  drawdown panel and a side-by-side benchmark table.
- **P&L distribution** — mean/median/std, the trade histogram with a normal fit, plus the
  **bootstrapped confidence interval** on expectancy and `P(> 0)`.
- **Fragility** — expectancy after dropping the best 1/3/5/10 trades; top-5% share of profit.
- **By period** — expectancy by year and by side.
- **Trades** — the raw table, downloadable.

---

## Reading the results

The headline equity curve is the least informative output. In rough order of what to trust:

1. **Expectancy net of realistic costs.** Use the slippage control — an intraday strategy that
   profits at zero cost and dies at 2¢/share was never viable. Estimate ~\$10–15 round trip on
   275 SPY shares (spread + stop-order slippage), then see whether the edge survives.
2. **Bootstrapped CI on expectancy.** If the lower bound sits near zero, you have "probably
   positive, magnitude unknown" — not a tradable finding.
3. **Fragility.** If dropping five trades from several hundred flips the strategy negative, the
   backtest measured a few lucky days.
4. **Out-of-sample survival.** Nothing in this pipeline can establish that the strategy works —
   only that it hasn't failed on data it was built on.

**On the statistics.** Trade P&L is skewed and fat-tailed by design, so the CI is bootstrapped
rather than `mean ± 1.96×SE` — the normality that formula needs is exactly what this data
lacks. And read it strictly: a 95% interval comes from a _procedure_ that captures the truth
95% of the time; it is **not** a 95% probability that the true value sits inside. Nor does it
account for **selection** — if you tried several filter variants and kept the best, the
interval is conditional on that choice and is optimistic. Only fresh data fixes that.

## Robustness checklist

Before believing anything here:

- [ ] **Parameter sensitivity** — vary ATR period and stop multiplier. You want a _plateau_, not
      a spike. A spike means you found a crevice in the noise.
- [ ] **Out-of-sample** — hold back a chronological block, look once.
- [ ] **Cross-instrument** — do the unchanged rules work on IWM, DIA, non-US indices?
- [ ] **Regime slicing** — is the edge present each year, or only in one stretch?
- [ ] **Monte Carlo on trade order** — your historical drawdown is one draw; shuffling usually
      reveals worse ones.
- [ ] **Cost stress** — double commissions, add slippage.
- [ ] **Random-filter test** — replace the context filter with a random one keeping the same
      fraction of trades, 1000×. If your filter doesn't beat that distribution, it isn't doing
      anything.
- [ ] **Synthetic data** — run on a random walk. It should make nothing. If it profits, there's
      a bug or a look-ahead leak.
