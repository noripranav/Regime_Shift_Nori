"""
Phase 1 -- Data pipeline for the Regime-Shift capstone.

Fetches daily prices for the Indian-market asset trio (equity / long
gilt bonds / gold) plus India VIX as an auxiliary regime-detection
feature, then builds one clean, aligned DataFrame of log returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf

# Portfolio assets: what we actually hold and reallocate between.
PRICE_TICKERS = {
    "equity": "NIFTYBEES.NS",    # Nippon India ETF Nifty 50 BeES
    "bonds":  "LTGILTBEES.NS",   # Nippon India ETF Nifty 8-13yr G-Sec Long Term Gilt
    "gold":   "GOLDBEES.NS",     # Nippon India ETF Gold BeES
}

# Auxiliary feature only -- NOT a portfolio holding, just a fear-gauge
# input to the HMM later.
VIX_TICKER = "^INDIAVIX"

# See build_master_dataset: no real single-day move for these assets
# comes remotely close to this. Anything past it is treated as bad
# data (bad tick / unadjusted corporate action), not a market move.
EXTREME_RETURN_CLIP = 0.25


def fetch_raw_prices(tickers, start, end=None):
    """
    Thin wrapper around yf.download. This is the ONLY function in this
    module that touches the network -- kept separate so the transform
    logic below (build_master_dataset) can be unit-tested without it.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    return raw


def build_master_dataset(raw, price_tickers, vix_ticker):
    """
    Turns yfinance's raw MultiIndex download into two aligned pieces:

    master: log returns for the tradable assets + the VIX level --
            what Phase 2 (volatility features) and the HMM consume.
    prices: raw aligned Close prices for the tradable assets only --
            needed for momentum (P_t / P_t-N - 1, defined on price
            levels, not returns) and for the regime-overlay plot
            deliverable, which needs actual price data under the
            colored regime bands, not a return series.

    raw:           DataFrame as returned by yf.download() for
                   price_tickers + [vix_ticker]
    price_tickers: list of tickers that are actual portfolio holdings
    vix_ticker:    the volatility-index ticker (feature only, never held)
    """
    close = raw["Close"].copy()
    close.columns.name = None

    # --- Portfolio assets: log returns, not simple returns ---
    # Log returns compound additively across time, which is what lets
    # us sum daily returns into a period return without a separate
    # compounding step later in the backtest.
    asset_prices = close[price_tickers]
    asset_returns = np.log(asset_prices).diff()

    # Report on the RAW data before touching it -- an honest diagnostic,
    # not a diagnostic on data that's already been silently altered.
    warn_on_data_quality_issues(asset_prices, asset_returns)

    # A single-day log return past EXTREME_RETURN_CLIP is not a real
    # NIFTY/gold/gilt move -- it's a bad tick or an unadjusted split/
    # bonus auto_adjust missed (confirmed on this project's real data:
    # NIFTYBEES.NS hit +230%, GOLDBEES.NS hit -461% on single days,
    # both clearly data errors, not market moves). Clipped here, at the
    # source, so it can't corrupt the HMM features OR backtest.py's
    # portfolio math downstream -- both consume asset_returns from this
    # one function.
    n_clipped = int((asset_returns.abs() > EXTREME_RETURN_CLIP).sum().sum())
    if n_clipped:
        print(f"Clipping {n_clipped} of those single-day move(s) to +-{EXTREME_RETURN_CLIP:.0%}.")
    asset_returns = asset_returns.clip(-EXTREME_RETURN_CLIP, EXTREME_RETURN_CLIP)

    # --- VIX: a LEVEL, not a return ---
    # India VIX is already an annualized expected-volatility reading, not
    # something you hold or trade. Return-transforming it would throw
    # away exactly the information the HMM wants (the level of fear
    # right now), so it stays as-is.
    vix_level = close[[vix_ticker]].rename(columns={vix_ticker: "VIX"})

    # --- Align on a shared calendar ---
    # inner join = keep only dates where EVERY series has a value. NSE
    # and the VIX index mostly share a calendar, but any stray gap (a
    # data-vendor hiccup, a late listing) gets dropped rather than
    # silently forward-filled, which would fabricate a return that
    # never happened.
    master = asset_returns.join(vix_level, how="inner").dropna()
    prices = asset_prices.loc[master.index]  # same trimmed, aligned index

    return master, prices


def warn_on_data_quality_issues(prices, log_returns, flat_run_threshold=5, extreme_return_threshold=0.20):
    """
    Flags (doesn't block) two concrete failure modes: a price frozen for
    several consecutive days (stale/bad tick from the data vendor), and
    a single-day |log return| big enough that it's more likely a data
    error (bad tick, unadjusted corporate action) than a real market
    move -- NIFTY's worst-ever single days are still well under 20%.
    Either one can silently wreck a zero-variance division downstream
    (z-scoring, vol features) or blow up a backtest's numbers.
    """
    for col in prices.columns:
        is_flat = prices[col].diff().eq(0)
        run_lengths = is_flat.groupby((~is_flat).cumsum()).cumsum()
        max_flat_run = int(run_lengths.max()) if len(run_lengths) else 0
        if max_flat_run >= flat_run_threshold:
            worst_end = run_lengths.idxmax()
            print(f"WARNING: {col} has a {max_flat_run}-day stretch of an unchanged price "
                  f"ending {worst_end.date()} -- looks like stale/bad data, not real trading.")

        extreme = log_returns[col][log_returns[col].abs() > extreme_return_threshold]
        if len(extreme):
            worst = extreme.abs().idxmax()
            print(f"WARNING: {col} has {len(extreme)} single-day move(s) over "
                  f"{extreme_return_threshold:.0%}, worst {log_returns[col][worst]:.1%} on "
                  f"{worst.date()} -- check it's a real move, not a bad tick or an unadjusted split.")


def load_market_data(start="2016-01-01", end=None):
    """End-to-end Phase 1 entry point: network fetch + transform."""
    tickers = list(PRICE_TICKERS.values()) + [VIX_TICKER]
    raw = fetch_raw_prices(tickers, start=start, end=end)
    return build_master_dataset(raw, list(PRICE_TICKERS.values()), VIX_TICKER)


if __name__ == "__main__":
    df, prices = load_market_data()
    print(df.head())
    print(f"\n{len(df)} trading days, {df.index.min().date()} to {df.index.max().date()}")