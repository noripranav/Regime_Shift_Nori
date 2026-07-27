"""
Offline test for build_master_dataset(). Uses synthetic price data shaped
exactly like a real yf.download() result (same MultiIndex column layout),
so it verifies the transform logic without ever touching the network.

Run: python test_data.py
"""

import numpy as np
import pandas as pd

from data import build_master_dataset, PRICE_TICKERS, VIX_TICKER


def make_fake_raw(n=600, seed=42):
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)

    equity = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.011, n)))
    bonds = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, n)))
    gold = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n)))
    vix = np.clip(18 + 8 * np.sin(np.linspace(0, 6, n)) + np.random.normal(0, 2, n), 9, 45)

    tickers = list(PRICE_TICKERS.values()) + [VIX_TICKER]
    fields = ["Close", "Open", "High", "Low", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers])
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    for tkr, series in zip(PRICE_TICKERS.values(), [equity, bonds, gold]):
        raw[("Close", tkr)] = series
    raw[("Close", VIX_TICKER)] = vix

    # Inject one stray gap on a single ticker, like a real vendor hiccup --
    # this is what exercises the inner-join-drops-the-date behavior.
    raw.loc[dates[50], ("Close", VIX_TICKER)] = np.nan
    return raw, n


def test_build_master_dataset():
    raw, n_calendar_rows = make_fake_raw()
    master, prices = build_master_dataset(raw, list(PRICE_TICKERS.values()), VIX_TICKER)

    # 1 row lost to the return calc (no prior price on day 1), 1 row lost
    # to the injected NaN gap.
    assert len(master) == n_calendar_rows - 2, "row count doesn't match expected drops"

    assert list(master.columns) == list(PRICE_TICKERS.values()) + ["VIX"], "unexpected columns"

    assert master.isna().sum().sum() == 0, "NaNs leaked through the join"

    # VIX must stay in a plausible level range, not get log-return-transformed
    # into something near zero.
    assert master["VIX"].between(5, 60).all(), "VIX doesn't look like a level anymore"

    # prices must be the same index as master (same trim), same tickers,
    # and actual price levels (positive, not centered near zero like a
    # return series would be).
    assert list(prices.index) == list(master.index), "prices/master index mismatch"
    assert list(prices.columns) == list(PRICE_TICKERS.values()), "unexpected price columns"
    assert (prices > 0).all().all(), "prices should never be zero/negative"

    print("All checks passed.")
    print(master.head())
    print(f"\n{len(master)} rows, {master.index.min().date()} to {master.index.max().date()}")


if __name__ == "__main__":
    test_build_master_dataset()
