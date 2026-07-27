"""
Phase 2 -- Feature engineering for the Regime-Shift capstone.

Builds momentum + volatility candidate features at multiple horizons.
Operates purely on the DataFrames from data.py -- no network access
needed. Which subset of these actually feeds the HMM is a Phase 3
decision (regime.py) -- this module just builds the candidate pool.
"""

import numpy as np
import pandas as pd

MOMENTUM_WINDOWS = [5, 21, 63, 126]   # ~1wk, 1mo, 1qtr, 6mo (trading days)
VOL_WINDOWS = [5, 21, 63]
ZSCORE_MIN_PERIODS = 63               # don't z-score off less than ~1 quarter of history
CLIP_SIGMA = 8.0                      # cap z-scores here -- see zscore_expanding docstring


def add_momentum_features(prices, windows=MOMENTUM_WINDOWS):
    """Momentum_N(t) = P_t / P_(t-N) - 1, off price levels directly."""
    out = pd.DataFrame(index=prices.index)
    for asset in prices.columns:
        for w in windows:
            out[f"{asset}_mom_{w}d"] = prices[asset].pct_change(w)
    return out


def add_volatility_features(returns, windows=VOL_WINDOWS):
    """Rolling realized volatility, annualized, off log returns."""
    out = pd.DataFrame(index=returns.index)
    for asset in returns.columns:
        for w in windows:
            out[f"{asset}_vol_{w}d"] = returns[asset].rolling(w).std() * np.sqrt(252)
    return out


def zscore_expanding(df, min_periods=ZSCORE_MIN_PERIODS):
    """
    Lookahead-safe z-score utility: mu/sigma at time t only use data up
    to and including t (expanding window), never the full sample.

    A column with ~zero variance (a stale/flat stretch in real data --
    a price that didn't move for several days) makes sigma ~0, which
    would otherwise silently divide out to inf and corrupt anything
    downstream. That's turned into NaN here instead -- same as any
    other missing data, gets caught by the caller's .dropna() rather
    than silently poisoning an HMM fit.

    Separately, clipped to +-CLIP_SIGMA: a single bad tick or unadjusted
    corporate action can produce a z-score in the hundreds, which
    overflows the Gaussian likelihood math inside hmmlearn (the
    "divide by zero encountered in matmul" warning). A z-score past 8
    is never real signal -- under an actual Gaussian it's a ~1-in-10^15
    event -- so capping it protects the fit without needing to first
    track down which specific day caused it.

    Not applied inside build_feature_matrix on purpose -- Phase 3 calls
    this on whichever specific features actually get fed to the HMM,
    since z-scoring only matters for what you're about to model, and
    doing it here on all ~20 candidate columns would be wasted work.
    """
    mu = df.expanding(min_periods=min_periods).mean()
    sigma = df.expanding(min_periods=min_periods).std()
    sigma = sigma.where(sigma > 1e-10)  # ~0 std -> NaN, not inf
    z = (df - mu) / sigma

    clipped = z.clip(-CLIP_SIGMA, CLIP_SIGMA)
    n_clipped = int((z != clipped).sum().sum())
    if n_clipped:
        print(f"NOTE: zscore_expanding clipped {n_clipped} value(s) past +-{CLIP_SIGMA} "
              f"std devs -- likely a bad tick or unadjusted split, not real signal.")
    return clipped


def build_feature_matrix(master, prices):
    """
    master: the (returns + VIX) DataFrame from data.build_master_dataset()
    prices: the aligned Close-price DataFrame from the same function

    Returns the RAW candidate feature pool: momentum (from prices) +
    volatility (from returns) at every window, plus the raw VIX level.
    Left un-z-scored on purpose -- these values need to stay
    interpretable for the Phase 2 sanity check (e.g. "does vol_21d
    actually spike around a known stressed period").
    """
    asset_cols = [c for c in master.columns if c != "VIX"]
    returns = master[asset_cols]
    aligned_prices = prices.loc[master.index, asset_cols]

    momentum = add_momentum_features(aligned_prices)
    volatility = add_volatility_features(returns)

    features = pd.concat([momentum, volatility, master[["VIX"]]], axis=1).dropna()
    return features


if __name__ == "__main__":
    from data import load_market_data

    master, prices = load_market_data()
    feats = build_feature_matrix(master, prices)
    print(feats.head())
    print(f"\n{len(feats)} rows, {feats.shape[1]} feature columns")