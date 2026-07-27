"""
Offline test for features.py. Builds a synthetic calm-then-crisis price
series (same idea as guide-notebook Section 7's leak demo: calm returns,
then a violent regime at the end) so we can check the volatility and
momentum features actually respond the way a real Crisis period should
make them respond -- before ever touching real NSE data.

Run: python test_features.py
"""

import numpy as np
import pandas as pd

from data import PRICE_TICKERS, VIX_TICKER
from features import build_feature_matrix, MOMENTUM_WINDOWS, VOL_WINDOWS


def make_calm_then_crisis(n_calm=400, n_crisis=40, seed=7):
    """
    Equity: calm drift, then a sharp selloff.
    Gold:   calm drift, then a mild flight-to-safety rally.
    Bonds:  calm drift, then a mild flight-to-quality rally.
    VIX:    low and steady, then spikes hard.
    """
    np.random.seed(seed)
    n = n_calm + n_crisis
    dates = pd.bdate_range("2019-01-01", periods=n)

    eq_ret = np.concatenate([
        np.random.normal(0.0005, 0.008, n_calm),
        np.random.normal(-0.012, 0.035, n_crisis),
    ])
    gold_ret = np.concatenate([
        np.random.normal(0.0002, 0.006, n_calm),
        np.random.normal(0.003, 0.012, n_crisis),
    ])
    bond_ret = np.concatenate([
        np.random.normal(0.0001, 0.002, n_calm),
        np.random.normal(0.0015, 0.004, n_crisis),
    ])
    vix = np.concatenate([
        np.random.normal(14, 1.5, n_calm),
        np.random.normal(38, 6, n_crisis),
    ])
    vix = np.clip(vix, 9, 80)

    tickers = list(PRICE_TICKERS.values())
    prices = pd.DataFrame({
        tickers[0]: 100 * np.exp(np.cumsum(eq_ret)),
        tickers[1]: 100 * np.exp(np.cumsum(bond_ret)),
        tickers[2]: 100 * np.exp(np.cumsum(gold_ret)),
    }, index=dates)

    returns = np.log(prices).diff()
    master = returns.copy()
    master["VIX"] = vix
    master = master.dropna()
    prices = prices.loc[master.index]

    crisis_start = dates[n_calm]
    return master, prices, crisis_start


def test_feature_shape_and_lookback():
    master, prices, _ = make_calm_then_crisis()
    feats = build_feature_matrix(master, prices)

    expected_cols = (
        len(PRICE_TICKERS) * len(MOMENTUM_WINDOWS)
        + len(PRICE_TICKERS) * len(VOL_WINDOWS)
        + 1  # VIX
    )
    assert feats.shape[1] == expected_cols, f"expected {expected_cols} feature columns, got {feats.shape[1]}"

    # The longest momentum window (126d) is the binding constraint on
    # how many early rows get dropped -- confirms dropna() isn't
    # silently eating more (or less) than it should.
    longest_window = max(MOMENTUM_WINDOWS)
    assert len(feats) == len(master) - longest_window, "unexpected row count after dropna"

    assert feats.isna().sum().sum() == 0, "NaNs leaked into the feature matrix"


def test_volatility_spikes_in_crisis():
    master, prices, crisis_start = make_calm_then_crisis()
    feats = build_feature_matrix(master, prices)

    equity_ticker = list(PRICE_TICKERS.values())[0]
    vol_col = f"{equity_ticker}_vol_21d"

    calm_vol = feats.loc[feats.index < crisis_start, vol_col].mean()
    crisis_vol = feats.loc[feats.index >= crisis_start, vol_col].mean()

    assert crisis_vol > calm_vol * 2, (
        f"21d vol didn't spike in the crisis window (calm={calm_vol:.3f}, "
        f"crisis={crisis_vol:.3f}) -- feature isn't picking up stress"
    )


def test_momentum_turns_negative_in_crisis():
    master, prices, crisis_start = make_calm_then_crisis()
    feats = build_feature_matrix(master, prices)

    equity_ticker = list(PRICE_TICKERS.values())[0]
    mom_col = f"{equity_ticker}_mom_21d"

    # A few weeks into the crisis, trailing 21d momentum should have
    # turned clearly negative.
    post_crisis_mom = feats.loc[feats.index >= crisis_start, mom_col].iloc[15:]
    assert (post_crisis_mom < 0).mean() > 0.8, "momentum didn't turn negative during the selloff"


def test_vix_stays_a_level():
    master, prices, _ = make_calm_then_crisis()
    feats = build_feature_matrix(master, prices)
    assert feats["VIX"].between(5, 90).all(), "VIX left its plausible level range"


if __name__ == "__main__":
    test_feature_shape_and_lookback()
    test_volatility_spikes_in_crisis()
    test_momentum_turns_negative_in_crisis()
    test_vix_stays_a_level()
    print("All checks passed.")

    master, prices, crisis_start = make_calm_then_crisis()
    feats = build_feature_matrix(master, prices)
    eq = list(PRICE_TICKERS.values())[0]
    print(f"\n21d vol, calm window mean:   {feats.loc[feats.index < crisis_start, f'{eq}_vol_21d'].mean():.3f}")
    print(f"21d vol, crisis window mean: {feats.loc[feats.index >= crisis_start, f'{eq}_vol_21d'].mean():.3f}")
