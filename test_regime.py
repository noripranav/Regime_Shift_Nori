"""
Offline test for regime.py. Builds a synthetic Bull -> Bear -> Crisis ->
Bull price series with known ground-truth regime windows, fits the HMM
on it, and checks the output is structurally sound and roughly tracks
the known regimes -- before ever touching real NSE data.

Run: python test_regime.py
"""

import numpy as np
import pandas as pd

from data import PRICE_TICKERS, VIX_TICKER
from features import build_feature_matrix
from regime import select_hmm_columns, fit_regime_model, label_states, transition_matrix_df, REGIME_NAMES
from features import zscore_expanding


def make_multi_regime_data(seed=11):
    """~4.5 years of synthetic data, four back-to-back windows with
    genuinely different drift/vol so the HMM has real structure to find."""
    np.random.seed(seed)
    windows = [
        ("bull",   500, 0.0007, 0.008, 14),
        ("bear",   250, -0.0006, 0.014, 22),
        ("crisis", 60,  -0.014, 0.040, 45),
        ("bull",   350, 0.0008, 0.009, 15),
    ]
    dates = pd.bdate_range("2017-01-02", periods=sum(w[1] for w in windows))

    eq_ret, gold_ret, bond_ret, vix, truth = [], [], [], [], []
    for name, n, mu, sigma, vix_level in windows:
        eq_ret.append(np.random.normal(mu, sigma, n))
        gold_ret.append(np.random.normal(0.0002, 0.007, n))
        bond_ret.append(np.random.normal(0.0001, 0.003, n))
        vix.append(np.clip(np.random.normal(vix_level, vix_level * 0.15, n), 8, 85))
        truth += [name] * n

    tickers = list(PRICE_TICKERS.values())
    prices = pd.DataFrame({
        tickers[0]: 100 * np.exp(np.cumsum(np.concatenate(eq_ret))),
        tickers[1]: 100 * np.exp(np.cumsum(np.concatenate(bond_ret))),
        tickers[2]: 100 * np.exp(np.cumsum(np.concatenate(gold_ret))),
    }, index=dates)

    returns = np.log(prices).diff()
    master = returns.copy()
    master["VIX"] = np.concatenate(vix)
    master = master.dropna()
    prices = prices.loc[master.index]
    truth = pd.Series(truth, index=dates).loc[master.index]

    return master, prices, truth


def test_transition_matrix_is_stochastic():
    master, prices, _ = make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    X = zscore_expanding(select_hmm_columns(feats, PRICE_TICKERS["equity"])).dropna()
    model = fit_regime_model(X)
    _, label_map, _ = label_states(model, X)

    tm = transition_matrix_df(model, label_map)
    assert tm.shape == (3, 3), "transition matrix should be 3x3"
    row_sums = tm.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"transition matrix rows must sum to 1, got {row_sums.tolist()}"
    assert (tm.values >= 0).all() and (tm.values <= 1).all(), "transition probabilities must be in [0, 1]"


def test_all_three_regimes_present():
    master, prices, _ = make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    X = zscore_expanding(select_hmm_columns(feats, PRICE_TICKERS["equity"])).dropna()
    model = fit_regime_model(X)
    _, _, labels = label_states(model, X)

    found = set(labels.unique())
    assert found == set(REGIME_NAMES), f"expected all 3 regimes, only found {found} -- possible state collapse"


def test_crisis_window_mostly_labeled_crisis():
    """Not a precise day-by-day match (that's what Phase 4 backtests
    against) -- just: does the model land on 'Crisis' for most of the
    days we know are the synthetic crisis window, more than chance?"""
    master, prices, truth = make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    X = zscore_expanding(select_hmm_columns(feats, PRICE_TICKERS["equity"])).dropna()
    model = fit_regime_model(X)
    _, _, labels = label_states(model, X)

    truth_aligned = truth.loc[X.index]
    crisis_days = truth_aligned[truth_aligned == "crisis"].index
    if len(crisis_days) == 0:
        return  # crisis window got trimmed by feature lookback -- nothing to check
    hit_rate = (labels.loc[crisis_days] == "Crisis").mean()
    assert hit_rate > 0.5, f"only {hit_rate:.0%} of known-crisis days labeled Crisis"


if __name__ == "__main__":
    test_transition_matrix_is_stochastic()
    test_all_three_regimes_present()
    test_crisis_window_mostly_labeled_crisis()
    print("All checks passed.")

    master, prices, truth = make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    X = zscore_expanding(select_hmm_columns(feats, PRICE_TICKERS["equity"])).dropna()
    model = fit_regime_model(X)
    _, label_map, labels = label_states(model, X)

    print("\nTransition matrix:\n", transition_matrix_df(model, label_map).round(3))
    print("\nRegime counts:\n", labels.value_counts())
    crisis_days = truth.loc[X.index]
    crisis_days = crisis_days[crisis_days == "crisis"].index
    if len(crisis_days):
        print(f"\nHit rate on known-crisis days: {(labels.loc[crisis_days] == 'Crisis').mean():.0%}")
