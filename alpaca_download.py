#!/usr/bin/env python3
"""
Alpaca Market Data Downloader.

DESCRIPTION:
    Downloads historical OHLCV (Open, High, Low, Close, Volume) bar data from Alpaca Markets
    and stores it in CSV format. Supports incremental updates - if a CSV file already exists,
    only missing data is fetched and appended.

DATA FORMAT:
    CSV files contain the following columns:
    - t  : Timestamp in ISO 8601 format (UTC timezone, e.g., 2024-01-02T09:00:00Z)
    - o  : Open price (decimal)
    - h  : High price (decimal)
    - l  : Low price (decimal)
    - c  : Close price (decimal)
    - v  : Volume (integer, number of shares)
    - n  : Number of trades (integer)
    - vw : Volume-weighted average price (decimal)

    All timestamps are in UTC timezone (indicated by 'Z' suffix).
    All price data is unadjusted (raw).

FEATURES:
    - Incremental downloads: appends only new data to existing files
    - Full history downloads: downloads from START_DATE when no file exists
    - Progress bars: shows real-time download progress for each ticker
    - Error handling: automatic retries with exponential backoff
    - Rate limiting: respects API rate limits automatically
    - SIP datafeed: uses Securities Information Processor for consolidated data

USAGE:
    1. Set required environment variables:
       export APCA-API-KEY-ID="your_key_id"
       export APCA-API-SECRET-KEY="your_secret_key"

    2. Configure tickers and parameters in the Config section below

    3. Run the script:
       python3 alpaca_download.py

REQUIREMENTS:
    - Python 3.7+
    - requests
    - tqdm
    - python-dateutil

ENVIRONMENT VARIABLES (required):
    APCA-API-KEY-ID     : Your Alpaca API key ID
    APCA-API-SECRET-KEY : Your Alpaca API secret key
"""

from dotenv import load_dotenv
import os
import csv
import time
import errno
import logging
from datetime import datetime, timezone, timedelta
from typing import Iterator, List, Dict, Optional

import requests
from tqdm import tqdm
from dateutil import parser as dtparser

# ============================================================================
# CONFIGURATION
# ============================================================================

# Tickers to download (add your symbols here)
TICKERS = ["SPY", "QQQ", "IWM", "GLD", "USO", "DIA"]

# Start date for initial downloads (when CSV doesn't exist)
# Format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ"
# Examples: "2024-01-01", "2023-06-15T00:00:00Z"
# Set to None to download maximum available history
START_DATE = "2015-01-01"

# Timeframe for bars (1Min, 5Min, 15Min, 1Hour, 1Day, etc.)
TIMEFRAME = "1Min"

# Target directory for CSV files
# CSV_DIR = "etfs-1min"
CSV_DIR = "market-data"

# ============================================================================
# ADVANCED SETTINGS (usually no need to change)
# ============================================================================

BASE_URL = "https://data.alpaca.markets/v2/stocks"  # Alpaca API base URL
DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"  # ISO 8601 datetime format
PAGE_LIMIT = 10000  # Max bars per API request
RETRY_MAX = 5  # Max retry attempts for failed requests
RETRY_BACKOFF = 2.0  # Exponential backoff base for retries
SESSION_TIMEOUT = 60  # HTTP request timeout in seconds


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utcnow_iso() -> str:
    """
    Get current UTC time in ISO 8601 format.
    Returns:
        String representation of current UTC time (e.g., "2024-01-02T15:30:00Z")
    """
    return datetime.now(timezone.utc).strftime(DATE_FMT)


def parse_start_date(date_str: Optional[str]) -> str:
    """
    Parse START_DATE configuration into ISO 8601 format.
    Args:
        date_str: Date string in "YYYY-MM-DD" or ISO format, or None
    Returns:
        ISO 8601 formatted datetime string in UTC
    Raises:
        ValueError: If date_str format is invalid
    """
    if date_str is None:
        # Return very early date for maximum available history
        return "1900-01-01T00:00:00Z"

    # Try to parse as ISO 8601 first
    try:
        dt = dtparser.isoparse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime(DATE_FMT)
    except Exception:
        # Fall back to simple YYYY-MM-DD format
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime(DATE_FMT)
        except Exception as e:
            raise ValueError(
                f"Invalid START_DATE format: {date_str}. "
                "Use 'YYYY-MM-DD' or ISO format."
            ) from e


def very_early_iso() -> str:
    """
    Return a very early timestamp for downloading maximum available history.
    Returns:
        ISO 8601 timestamp string representing 1900-01-01
    """
    return "1900-01-01T00:00:00Z"


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.
    Args:
        path: Directory path to create
    Raises:
        OSError: If directory creation fails for reasons other than already existing
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def csv_path_for(symbol: str) -> str:
    """
    Generate CSV file path for a given ticker symbol.
    Args:
        symbol: Stock ticker symbol (e.g., "SPY")
    Returns:
        Full path to CSV file (e.g., "etfs-1min/SPY.csv")
    """
    return os.path.join(CSV_DIR, f"{symbol.upper()}.csv")


def read_last_timestamp(csv_path: str) -> Optional[str]:
    """
    Read the timestamp of the last bar from an existing CSV file.
    This is used to determine where to resume downloading data.
    Args:
        csv_path: Path to the CSV file
    Returns:
        ISO timestamp string of last bar, or None if file doesn't exist or is empty
    """
    if not os.path.isfile(csv_path):
        return None

    # Read file and find the last non-empty line
    last_line = None
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line

    # Check if we have valid data (not just header)
    if not last_line or last_line.lower().startswith("t,"):
        return None

    # Extract timestamp from first column
    return last_line.strip().split(",")[0]


def iso_add_minute(iso_ts: str) -> str:
    """
    Add one minute to an ISO 8601 timestamp.
    This is used to start fetching from the bar immediately after the last downloaded bar.
    Args:
        iso_ts: ISO 8601 timestamp string
    Returns:
        ISO timestamp string incremented by one minute
    """
    dt = dtparser.isoparse(iso_ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt + timedelta(minutes=1)).astimezone(timezone.utc).strftime(DATE_FMT)


# ============================================================================
# API INTERACTION
# ============================================================================

def api_headers() -> Dict[str, str]:
    """
    Build HTTP headers for Alpaca API authentication.
    Reads API credentials from environment variables.
    Returns:
        Dictionary of HTTP headers with authentication credentials
    Raises:
        RuntimeError: If required environment variables are not set
    """
    load_dotenv()
    key = os.getenv("APCA-API-KEY-ID")
    secret = os.getenv("APCA-API-SECRET-KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing API credentials. Please set APCA-API-KEY-ID and "
            "APCA-API-SECRET-KEY in your environment."
        )

    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
    }


def fetch_bars(
        session: requests.Session,
        symbol: str,
        start_iso: str,
        end_iso: str,
        feed: str = "sip",
        adjustment: str = "raw",
) -> Iterator[List[Dict]]:
    """
    Fetch historical bar data from Alpaca API with pagination support.

    This generator function handles:
    - Pagination through large datasets
    - Rate limiting (429 responses)
    - Network errors with automatic retries
    - Exponential backoff on failures

    Args:
        session: Requests session with authentication headers
        symbol: Stock ticker symbol
        start_iso: Start timestamp in ISO 8601 format
        end_iso: End timestamp in ISO 8601 format
        feed: Data feed to use ("sip" for consolidated data)
        adjustment: Price adjustment type ("raw", "split", "dividend", "all")

    Yields:
        Lists of bar dictionaries, one list per API page

    Raises:
        RuntimeError: If maximum retries exceeded or API returns error
    """
    url = f"{BASE_URL}/{symbol}/bars"
    params = {
        "timeframe": TIMEFRAME,
        "start": start_iso,
        "end": end_iso,
        "limit": PAGE_LIMIT,
        "adjustment": adjustment,
        "feed": feed,
    }

    next_token = None
    attempt = 0

    while True:
        # Add pagination token if we're on a subsequent page
        if next_token:
            params["page_token"] = next_token
        else:
            params.pop("page_token", None)

        try:
            resp = session.get(url, params=params, timeout=SESSION_TIMEOUT)
        except requests.RequestException as e:
            # Network error - retry with exponential backoff
            if attempt < RETRY_MAX:
                time.sleep(RETRY_BACKOFF ** attempt)
                attempt += 1
                continue
            raise RuntimeError(f"Network error after {RETRY_MAX} retries: {e}") from e

        # Handle rate limiting (HTTP 429)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            time.sleep(max(retry_after, 1.0))
            continue

        # Handle API errors
        if resp.status_code >= 400:
            try:
                payload = resp.json()
            except Exception:
                payload = resp.text
            raise RuntimeError(f"API error {resp.status_code}: {payload}")

        # Parse response
        payload = resp.json()
        bars = payload.get("bars", [])

        # Yield bars if we got any
        if bars:
            yield bars

        # Check for more pages
        next_token = payload.get("next_page_token")
        if not next_token:
            break  # No more pages, we're done


# ============================================================================
# DATA STORAGE
# ============================================================================

def write_bars_to_csv(
        csv_path: str,
        bars: List[Dict],
        write_header_if_new: bool
) -> int:
    """
    Append bar data to CSV file.

    Args:
        csv_path: Path to CSV file
        bars: List of bar dictionaries from API
        write_header_if_new: Whether to write CSV header if file is new

    Returns:
        Number of bars written
    """
    wrote = 0
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header for new files
        if write_header_if_new and not file_exists:
            writer.writerow(["t", "o", "h", "l", "c", "v", "n", "vw"])

        # Write each bar as a row
        for b in bars:
            writer.writerow([
                b.get("t"),  # timestamp
                b.get("o"),  # open
                b.get("h"),  # high
                b.get("l"),  # low
                b.get("c"),  # close
                b.get("v"),  # volume
                b.get("n"),  # number of trades
                b.get("vw"),  # volume-weighted average price
            ])
            wrote += 1

    return wrote


# ============================================================================
# DOWNLOAD ORCHESTRATION
# ============================================================================

def download_symbol(symbol: str) -> None:
    """
    Download bar data for a single ticker symbol.
    This function handles both full downloads (when CSV doesn't exist) and
    incremental updates (appending new bars to existing CSV).
    Args:
        symbol: Stock ticker symbol to download
    Raises:
        RuntimeError: If API requests fail after retries
    """
    csv_path = csv_path_for(symbol)
    ensure_dir(CSV_DIR)

    # Check if we have existing data
    last_ts = read_last_timestamp(csv_path)

    if last_ts:
        # Append mode: continue from last timestamp
        start_iso = iso_add_minute(last_ts)
        mode = "append"
        print(f"\n{symbol}: APPEND MODE")
        print(f"  Last data timestamp: {last_ts}")
        print(f"  Fetching from: {start_iso}")
    else:
        # Full download mode: start from configured start date
        start_iso = parse_start_date(START_DATE)
        mode = "full"
        print(f"\n{symbol}: FULL DOWNLOAD")
        print(f"  Starting from: {start_iso}")

    # end_iso = utcnow_iso()
    end_iso = (
    datetime.now(timezone.utc) - timedelta(minutes=16)
).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"  Ending at: {end_iso}")
    print(f"  Output: {csv_path}")

    # Set up authenticated session
    headers = api_headers()
    session = requests.Session()
    session.headers.update(headers)

    # Create progress bar
    progress = tqdm(
        desc=f"{symbol} ({mode})",
        unit="bars",
        dynamic_ncols=True,
        leave=False,
    )

    total_written = 0
    first_bar_ts = None
    last_bar_ts = None

    try:
        # Fetch and write bars page by page
        for page in fetch_bars(session, symbol, start_iso, end_iso, feed="sip"):
            # Filter out bars we already have (in append mode)
            if last_ts:
                page = [b for b in page if b.get("t") and b["t"] > last_ts]

            if not page:
                progress.update(0)
                continue

            # Track timestamp range of downloaded data
            if not first_bar_ts and page:
                first_bar_ts = page[0].get("t")
            if page:
                last_bar_ts = page[-1].get("t")

            # Write bars to CSV
            wrote = write_bars_to_csv(
                csv_path,
                page,
                write_header_if_new=(mode == "full" and total_written == 0)
            )
            total_written += wrote
            progress.update(wrote)

    finally:
        progress.close()
        session.close()

    # Print summary
    if total_written == 0:
        print(f"  Result: Up to date (no new bars)")
    else:
        print(f"  Result: Wrote {total_written} new bars")
        if first_bar_ts and last_bar_ts:
            print(f"  Data range: {first_bar_ts} to {last_bar_ts}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main entry point for the script.
    Iterates through all configured tickers and downloads data for each.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Print configuration summary
    print("=" * 60)
    print("Alpaca Market Data Downloader")
    print("=" * 60)
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Output directory: {CSV_DIR}")
    print(f"Default start date: {START_DATE if START_DATE else 'Maximum available history'}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print("=" * 60)

    # Verify API credentials are set
    try:
        _ = api_headers()
    except RuntimeError as e:
        print(str(e))
        return

    # Download each ticker
    for sym in tqdm(TICKERS, desc="Tickers", dynamic_ncols=True):
        try:
            download_symbol(sym)
        except Exception as e:
            print(f"\n{sym}: ERROR -> {str(e)}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()