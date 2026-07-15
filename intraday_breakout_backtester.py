# v.003 - Intraday Breakout Strategy Backtester
# Description: Backtester for volatility-based breakout strategies with configurable parameters

import os
import pandas as pd
import numpy as np
from datetime import time as dt_time
import glob
import time as tm
from tqdm import tqdm

# ============================================================================
# CONFIGURATION SECTION - Modify these settings for your needs
# ============================================================================

CONFIG = {
    # --- DATA DIRECTORIES ---
    # Directory containing your input CSV files with 1-minute price data
    'data_dir': '/Users/mgianino/Documents/Projects/Trading-Volatility-Breakouts/market-data',

    # Directory where daily aggregated data will be saved
    'rt_data_dir': 'RT_data',

    # Directory where individual ticker trade files will be saved
    'rt_trades_dir': 'RT_trades',

    # Directory where individual ticker event files (ATR values) will be saved
    'rt_events_dir': 'RT_events',

    # --- STRATEGY PARAMETERS ---
    # Strategy name (used in trade records)
    'strategy_name': 'BRK',

    # Number of days for ATR (Average True Range) calculation
    'atr_period': 5,

    # Multiplier for calculating breakout levels from open price
    # Long breakout = Open + (breakout_multiplier * ATR)
    # Short breakout = Open - (breakout_multiplier * ATR)
    'breakout_multiplier': 0.33,

    # Multiplier for calculating stop loss distance from entry price
    # Long stop = Entry - (stoploss_multiplier * ATR)
    # Short stop = Entry + (stoploss_multiplier * ATR)
    'stoploss_multiplier': 0.33,

    # --- POSITION SIZING ---
    # Number of shares/contracts to trade per signal
    'qty': 100,

    # --- TIME FILTERS ---
    # Latest time to enter a new trade (HH, MM format in 24-hour time)
    # No new entries will be taken after this time
    'max_entry_time': (15, 0),  # 15:00 = 3:00 PM

    # Time to exit all positions at end of day (HH, MM format)
    # All open positions will be closed at this time
    'eod_exit_time': (16, 0),  # 16:00 = 4:00 PM

    # --- TRADE DIRECTION SETTINGS ---
    # If True: Can take both long AND short trades on the same day
    # If False: Only takes the FIRST signal of the day (either long or short)
    'both_directions': False,
}


# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


class Backtester:
    def __init__(self, data_dir='data', rt_data_dir='RT_data', rt_trades_dir='RT_trades',
                 rt_events_dir='RT_events', strategy_name='BRK', atr_period=3,
                 breakout_multiplier=0.3, stoploss_multiplier=0.3, qty=100,
                 max_entry_time=dt_time(14, 0), eod_exit_time=dt_time(15, 55),
                 both_directions=True):
        """
        Initialize the backtester with configuration parameters.

        Args:
            data_dir (str): Directory containing input data files
            rt_data_dir (str): Directory for exporting daily data
            rt_trades_dir (str): Directory for exporting trade data
            rt_events_dir (str): Directory for exporting ATR events
            strategy_name (str): Name of strategy (used in trade records)
            atr_period (int): Period for ATR calculation
            breakout_multiplier (float): Multiplier for breakout level calculation
            stoploss_multiplier (float): Multiplier for stop loss calculation
            qty (int): Quantity of shares to trade
            max_entry_time (datetime.time): Latest time to enter a trade
            eod_exit_time (datetime.time): Time to exit trades at end of day
            both_directions (bool): Whether to take both long and short trades on same day
        """
        # Validate configuration
        self._validate_config(atr_period, breakout_multiplier, stoploss_multiplier,
                              qty, max_entry_time, eod_exit_time)

        self.data_dir = data_dir
        self.rt_data_dir = rt_data_dir
        self.rt_trades_dir = rt_trades_dir
        self.rt_events_dir = rt_events_dir
        self.strategy_name = strategy_name
        self.atr_period = atr_period
        self.breakout_multiplier = breakout_multiplier
        self.stoploss_multiplier = stoploss_multiplier
        self.qty = qty
        self.max_entry_time = max_entry_time
        self.eod_exit_time = eod_exit_time
        self.both_directions = both_directions

        # Create output directories if they don't exist
        for directory in [rt_data_dir, rt_trades_dir, rt_events_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _validate_config(self, atr_period, breakout_multiplier, stoploss_multiplier,
                         qty, max_entry_time, eod_exit_time):
        """Validate configuration parameters"""
        if atr_period < 1:
            raise ValueError("ATR period must be at least 1")
        if breakout_multiplier <= 0:
            raise ValueError("Breakout multiplier must be positive")
        if stoploss_multiplier <= 0:
            raise ValueError("Stop loss multiplier must be positive")
        if qty <= 0:
            raise ValueError("Quantity must be positive")
        if max_entry_time >= eod_exit_time:
            raise ValueError("Max entry time must be before EOD exit time")

        # Check times are within trading hours
        market_open = dt_time(9, 30)
        market_close = dt_time(16, 0)
        if not (market_open <= max_entry_time <= market_close):
            raise ValueError(f"Max entry time must be between {market_open} and {market_close}")
        if not (market_open <= eod_exit_time <= market_close):
            raise ValueError(f"EOD exit time must be between {market_open} and {market_close}")

    def load_data(self, file_path):
        """
        Load 1-minute data from a file and convert to Eastern time.

        Args:
            file_path (str): Path to the data file

        Returns:
            pd.DataFrame: DataFrame containing the loaded data
            str: Ticker symbol
        """
        # Extract ticker from filename (e.g., "QQQ.csv" -> "QQQ")
        ticker = os.path.basename(file_path).replace('.csv', '')

        # Load the data
        df = pd.read_csv(file_path)

        # Parse timestamp as UTC and convert to Eastern time
        df['datetime'] = pd.to_datetime(df['t'], utc=True).dt.tz_convert('US/Eastern')

        # Remove timezone info to make it timezone-naive
        df['datetime'] = df['datetime'].dt.tz_localize(None)

        # Filter for Regular Trading Hours (09:30 - 16:00)
        df = df.set_index('datetime')
        df = df.between_time('09:30', '16:00')
        df = df.reset_index()

        # Rename columns to standard format
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })

        # Add ticker column
        df['ticker'] = ticker

        # Select only the columns we need
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'ticker']]

        return df, ticker

    def convert_to_daily(self, df):
        """
        Convert 1-minute data to daily bars (OHLCV).

        Args:
            df (pd.DataFrame): DataFrame with 1-minute bar data

        Returns:
            pd.DataFrame: DataFrame with daily OHLCV data
        """
        # Create a date column
        df['date'] = df['datetime'].dt.date

        # Group by date and aggregate
        daily_df = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'ticker': 'first'
        }).reset_index()

        # Convert the date column to datetime
        daily_df['date'] = pd.to_datetime(daily_df['date'])

        return daily_df

    def calculate_atr(self, df, period=3):
        """
        Calculate ATR (Average True Range) for the given dataframe.

        Args:
            df (pd.DataFrame): DataFrame with price data
            period (int): Period for ATR calculation

        Returns:
            pd.Series: Series with ATR values
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum.reduce([high_low, high_close, low_close])
        true_range_series = pd.Series(true_range, index=df.index)
        atr = true_range_series.rolling(window=period).mean()
        return atr

    def process_long_trade(self, ticker, entry_time, entry_bar, long_breakout, long_stop,
                           eod_bar, day_minute_data):
        """
        Process a long trade and return the trade details.

        Args:
            ticker (str): Ticker symbol
            entry_time (datetime): Entry time for the trade
            entry_bar (pd.Series): Bar data where entry occurred
            long_breakout (float): Long breakout price
            long_stop (float): Long stop loss price
            eod_bar (pd.Series): End of day bar for exits
            day_minute_data (pd.DataFrame): All minute data for the day

        Returns:
            dict: Trade details
        """
        # Find exit point (either stop-loss or EOD)
        remaining_bars = day_minute_data.loc[day_minute_data['datetime'] > entry_time]

        # Check for stop loss
        stop_hit = False
        if not remaining_bars.empty:
            stop_bars = remaining_bars[remaining_bars['low'] <= long_stop]
            if not stop_bars.empty:
                exit_bar = stop_bars.iloc[0]
                exit_time = exit_bar['datetime']
                exit_price = long_stop
                stop_hit = True

        # If no stop loss hit, exit at EOD
        if not stop_hit:
            exit_time = eod_bar['datetime']
            exit_price = eod_bar['close']

        return {
            'Symbol': ticker,
            'Strategy': self.strategy_name,
            'Side': 'Long',
            'DateIn': entry_time,
            'QtyIn': self.qty,
            'PriceIn': long_breakout,
            'DateOut': exit_time,
            'QtyOut': self.qty,
            'PriceOut': exit_price,
            'FeesIn': 0,
            'FeesOut': 0
        }

    def process_short_trade(self, ticker, entry_time, entry_bar, short_breakout, short_stop,
                            eod_bar, day_minute_data):
        """
        Process a short trade and return the trade details.

        Args:
            ticker (str): Ticker symbol
            entry_time (datetime): Entry time for the trade
            entry_bar (pd.Series): Bar data where entry occurred
            short_breakout (float): Short breakout price
            short_stop (float): Short stop loss price
            eod_bar (pd.Series): End of day bar for exits
            day_minute_data (pd.DataFrame): All minute data for the day

        Returns:
            dict: Trade details
        """
        # Find exit point (either stop-loss or EOD)
        remaining_bars = day_minute_data.loc[day_minute_data['datetime'] > entry_time]

        # Check for stop loss
        stop_hit = False
        if not remaining_bars.empty:
            stop_bars = remaining_bars[remaining_bars['high'] >= short_stop]
            if not stop_bars.empty:
                exit_bar = stop_bars.iloc[0]
                exit_time = exit_bar['datetime']
                exit_price = short_stop
                stop_hit = True

        # If no stop loss hit, exit at EOD
        if not stop_hit:
            exit_time = eod_bar['datetime']
            exit_price = eod_bar['close']

        return {
            'Symbol': ticker,
            'Strategy': self.strategy_name,
            'Side': 'Short',
            'DateIn': entry_time,
            'QtyIn': self.qty,
            'PriceIn': short_breakout,
            'DateOut': exit_time,
            'QtyOut': self.qty,
            'PriceOut': exit_price,
            'FeesIn': 0,
            'FeesOut': 0
        }

    def run_breakout_strategy(self, minute_data, daily_df, ticker):
        """
        Run breakout strategy on the data.

        Args:
            minute_data (pd.DataFrame): DataFrame with 1-minute bar data
            daily_df (pd.DataFrame): DataFrame with daily OHLCV data
            ticker (str): Ticker symbol

        Returns:
            pd.DataFrame: DataFrame with trade results
            pd.DataFrame: DataFrame with ATR events
        """
        start_time = tm.time()

        # Calculate ATR
        daily_df['atr'] = round(self.calculate_atr(daily_df, self.atr_period), 4)
        # Shift ATR by 1 day to use yesterday's ATR for today's calculations
        daily_df['prev_day_atr'] = daily_df['atr'].shift(1)
        # Skip first few days where ATR or shifted ATR is NaN
        daily_df = daily_df.dropna(subset=['prev_day_atr']).copy()

        # Prepare events dataframe for ATR values
        events_df = pd.DataFrame({
            'Symbol': ticker,
            'Date': daily_df['date'].dt.strftime('%m/%d/%Y'),
            'Type': 300,
            'Value': daily_df['prev_day_atr']
        })

        # Initialize trades list
        trades = []

        # Pre-calculate breakout levels for all days
        daily_df.loc[:, 'long_breakout'] = round(
            daily_df['open'] + self.breakout_multiplier * daily_df['prev_day_atr'], 2)
        daily_df.loc[:, 'short_breakout'] = round(
            daily_df['open'] - self.breakout_multiplier * daily_df['prev_day_atr'], 2)
        daily_df.loc[:, 'long_stop'] = round(
            daily_df['long_breakout'] - self.stoploss_multiplier * daily_df['prev_day_atr'], 2)
        daily_df.loc[:, 'short_stop'] = round(
            daily_df['short_breakout'] + self.stoploss_multiplier * daily_df['prev_day_atr'], 2)

        # Create date and time columns for faster lookups
        minute_data['date'] = minute_data['datetime'].dt.date
        minute_data['time'] = minute_data['datetime'].dt.time

        # Process each day with progress bar (leave=False clears it after completion)
        for _, day_data in tqdm(daily_df.iterrows(), total=len(daily_df),
                                desc=f"  Processing {ticker}", leave=False,
                                position=1, ncols=80):
            date = day_data['date'].date()

            # Extract minute data for this day
            day_minute_data = minute_data[minute_data['date'] == date]

            if day_minute_data.empty:
                continue

            # Find EOD exit bar
            eod_bars = day_minute_data[day_minute_data['time'] >= self.eod_exit_time]
            if not eod_bars.empty:
                eod_bar = eod_bars.iloc[0]
            else:
                eod_bar = day_minute_data.iloc[-1]

            # Filter for bars before max entry time
            entry_bars = day_minute_data[day_minute_data['time'] <= self.max_entry_time]
            if entry_bars.empty:
                continue

            # Get breakout values
            long_breakout = day_data['long_breakout']
            short_breakout = day_data['short_breakout']
            long_stop = day_data['long_stop']
            short_stop = day_data['short_stop']

            # Quick check if any breakout happened
            day_high = entry_bars['high'].max()
            day_low = entry_bars['low'].min()

            if not self.both_directions:
                # Only take the first signal of the day
                potential_long_breakout = day_high >= long_breakout
                potential_short_breakout = day_low <= short_breakout

                if potential_long_breakout or potential_short_breakout:
                    first_long_time = None
                    first_short_time = None
                    first_long_bar = None
                    first_short_bar = None

                    # Find first long breakout
                    if potential_long_breakout:
                        for idx, row in entry_bars.iterrows():
                            if row['high'] >= long_breakout:
                                first_long_time = row['datetime']
                                first_long_bar = row
                                break

                    # Find first short breakout
                    if potential_short_breakout:
                        for idx, row in entry_bars.iterrows():
                            if row['low'] <= short_breakout:
                                first_short_time = row['datetime']
                                first_short_bar = row
                                break

                    # Take the trade that happened first
                    if first_long_time is not None and first_short_time is not None:
                        if first_long_time <= first_short_time:
                            trade = self.process_long_trade(ticker, first_long_time, first_long_bar,
                                                            long_breakout, long_stop, eod_bar, day_minute_data)
                            trades.append(trade)
                        else:
                            trade = self.process_short_trade(ticker, first_short_time, first_short_bar,
                                                             short_breakout, short_stop, eod_bar, day_minute_data)
                            trades.append(trade)
                    elif first_long_time is not None:
                        trade = self.process_long_trade(ticker, first_long_time, first_long_bar,
                                                        long_breakout, long_stop, eod_bar, day_minute_data)
                        trades.append(trade)
                    elif first_short_time is not None:
                        trade = self.process_short_trade(ticker, first_short_time, first_short_bar,
                                                         short_breakout, short_stop, eod_bar, day_minute_data)
                        trades.append(trade)
            else:
                # Can take both long and short on same day
                if day_high >= long_breakout:
                    for idx, row in entry_bars.iterrows():
                        if row['high'] >= long_breakout:
                            trade = self.process_long_trade(ticker, row['datetime'], row,
                                                            long_breakout, long_stop, eod_bar, day_minute_data)
                            trades.append(trade)
                            break

                if day_low <= short_breakout:
                    for idx, row in entry_bars.iterrows():
                        if row['low'] <= short_breakout:
                            trade = self.process_short_trade(ticker, row['datetime'], row,
                                                             short_breakout, short_stop, eod_bar, day_minute_data)
                            trades.append(trade)
                            break

        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
            'Symbol', 'Strategy', 'Side', 'DateIn', 'QtyIn', 'PriceIn',
            'DateOut', 'QtyOut', 'PriceOut', 'FeesIn', 'FeesOut'
        ])

        end_time = tm.time()
        # Use tqdm.write to print without interfering with progress bars
        tqdm.write(f"  {ticker}: {end_time - start_time:.2f}s, {len(trades_df)} trades")

        return trades_df, events_df

    def export_data(self, daily_df, trades_df, events_df, ticker):
        """
        Export data to the specified directories.

        Args:
            daily_df (pd.DataFrame): DataFrame with daily data
            trades_df (pd.DataFrame): DataFrame with trade data
            events_df (pd.DataFrame): DataFrame with ATR events
            ticker (str): Ticker symbol
        """
        # Ensure output directories exist
        for directory in [self.rt_data_dir, self.rt_trades_dir, self.rt_events_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        # Export daily data
        daily_export = daily_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        daily_export.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        daily_export.to_csv(os.path.join(self.rt_data_dir, f"{ticker}.csv"), index=False)

        # Export trades
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(self.rt_trades_dir, f"{ticker}_trades.csv"), index=False)

        # Export ATR events
        events_df.to_csv(os.path.join(self.rt_events_dir, f"{ticker}_events.csv"), index=False)

    def combine_csv_files(self, input_dir, output_filename):
        """
        Combine all CSV files in the given directory into a single output file.

        Args:
            input_dir (str): Directory containing CSV files to combine
            output_filename (str): Name of the output combined file

        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            return False

        output_file = os.path.join(input_dir, output_filename)
        all_csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
        csv_files = [f for f in all_csv_files if os.path.basename(f) != output_filename]

        if not csv_files:
            print(f"No CSV files found to combine in {input_dir}")
            return False

        print(f"Combining {len(csv_files)} CSV files...")

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception as e:
                print(f"  Error reading {os.path.basename(csv_file)}: {e}")

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df.to_csv(output_file, index=False)
            print(f"✓ Combined into {output_filename} ({len(combined_df)} total rows)")
            return True
        else:
            print(f"No data to combine for {output_filename}")
            return False

    def combine_all_results(self):
        """
        Combine all individual CSV files into master files
        """
        print("\n" + "=" * 60)
        print("COMBINING RESULTS INTO MASTER FILES")
        print("=" * 60 + "\n")

        self.combine_csv_files(self.rt_trades_dir, f"{self.strategy_name}_trades.csv")
        self.combine_csv_files(self.rt_events_dir, f"{self.strategy_name}_events.csv")

        print("\n" + "=" * 60)
        print("RESULTS COMBINED SUCCESSFULLY!")
        print("=" * 60)

    def process_file(self, file_path):
        """
        Process a single data file.

        Args:
            file_path (str): Path to the data file

        Returns:
            tuple: (daily_df, trades_df, events_df) or None if error
        """
        try:
            # Load data
            minute_data, ticker = self.load_data(file_path)

            # Convert to daily
            daily_df = self.convert_to_daily(minute_data)

            # Run strategy
            trades_df, events_df = self.run_breakout_strategy(minute_data, daily_df, ticker)

            # Export results
            self.export_data(daily_df, trades_df, events_df, ticker)

            return daily_df, trades_df, events_df

        except Exception as e:
            tqdm.write(f"ERROR processing {os.path.basename(file_path)}: {e}")
            return None

    def run_all(self):
        """
        Process all data files in the data directory and combine results.
        """
        file_pattern = os.path.join(self.data_dir, "*.csv")
        files = glob.glob(file_pattern)

        if not files:
            print(f"No CSV files found in {self.data_dir}")
            return

        print(f"Found {len(files)} files to process\n")
        total_start_time = tm.time()

        successful = 0
        failed = 0

        for file_path in tqdm(files, desc="Processing files", position=0, ncols=80):
            result = self.process_file(file_path)
            if result is not None:
                successful += 1
            else:
                failed += 1

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {successful} successful, {failed} failed")
        print(f"Total time: {tm.time() - total_start_time:.2f} seconds")
        print(f"{'=' * 60}")

        # Combine all results into master files
        if successful > 0:
            self.combine_all_results()

    def print_configuration(self):
        """Print current configuration settings"""
        print("=" * 60)
        print("BACKTESTER CONFIGURATION")
        print("=" * 60)
        print(f"Data Directory:          {self.data_dir}")
        print(f"Output Directories:")
        print(f"  - Daily Data:          {self.rt_data_dir}")
        print(f"  - Trades:              {self.rt_trades_dir}")
        print(f"  - Events:              {self.rt_events_dir}")
        print(f"\nStrategy Parameters:")
        print(f"  - Strategy Name:       {self.strategy_name}")
        print(f"  - ATR Period:          {self.atr_period} days")
        print(f"  - Breakout Multiplier: {self.breakout_multiplier}")
        print(f"  - StopLoss Multiplier: {self.stoploss_multiplier}")
        print(f"  - Position Size:       {self.qty} shares")
        print(f"\nTime Filters:")
        print(f"  - Max Entry Time:      {self.max_entry_time.strftime('%H:%M')}")
        print(f"  - EOD Exit Time:       {self.eod_exit_time.strftime('%H:%M')}")
        print(f"\nTrade Direction:")
        print(f"  - Both Directions:     {self.both_directions}")
        direction_note = "Can take long AND short on same day" if self.both_directions else "Only FIRST signal of the day"
        print(f"    ({direction_note})")
        print("=" * 60 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    total_start = tm.time()

    # Create backtester with configuration from CONFIG dictionary
    backtester = Backtester(
        data_dir=CONFIG['data_dir'],
        rt_data_dir=CONFIG['rt_data_dir'],
        rt_trades_dir=CONFIG['rt_trades_dir'],
        rt_events_dir=CONFIG['rt_events_dir'],
        strategy_name=CONFIG['strategy_name'],
        atr_period=CONFIG['atr_period'],
        breakout_multiplier=CONFIG['breakout_multiplier'],
        stoploss_multiplier=CONFIG['stoploss_multiplier'],
        qty=CONFIG['qty'],
        max_entry_time=dt_time(*CONFIG['max_entry_time']),
        eod_exit_time=dt_time(*CONFIG['eod_exit_time']),
        both_directions=CONFIG['both_directions']
    )

    # Print configuration before starting
    backtester.print_configuration()

    # Process all files
    backtester.run_all()

    # Or process a single file (uncomment to use):
    # backtester.process_file("path/to/QQQ.csv")

    total_time = tm.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"BACKTESTING COMPLETED IN {total_time:.2f} SECONDS!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()