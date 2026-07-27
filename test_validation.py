"""
Offline test for validation.py. The most important check here isn't
shape/sanity -- it's test_truncating_future_data_does_not_change_past_folds,
which directly empirically verifies the property the whole project is
about: nothing past-fold ever changes if you feed the model MORE future
data. If that test fails, there's a real lookahead leak somewhere.

Run: python test_validation.py
"""

import numpy as np
import pandas as pd

from data import PRICE_TICKERS
from features import build_feature_matrix
from validation import walk_forward_splits, run_walk_forward_regimes
import test_regime


def test_splits_never_look_ahead():
    splits = walk_forward_splits(n_obs=1000, min_train_size=504, test_size=63, mode="expanding")
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "train indices must all precede test indices"
    # test windows should tile forward with no gaps and no overlaps
    for (_, test_a), (_, test_b) in zip(splits, splits[1:]):
        assert test_b.min() == test_a.max() + 1, "test windows should be contiguous, no gaps/overlaps"


def test_rolling_mode_keeps_fixed_train_size():
    splits = walk_forward_splits(n_obs=1000, min_train_size=300, test_size=63, mode="rolling")
    for train_idx, _ in splits[1:-1]:  # skip first/last, edge effects expected
        assert len(train_idx) == 300, "rolling window training size should stay fixed"


def test_walk_forward_runs_on_synthetic_data():
    master, prices, truth = test_regime.make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    labels, folds = run_walk_forward_regimes(
        feats, PRICE_TICKERS["equity"], min_train_size=300, test_size=63
    )
    assert len(folds) > 3
    assert len(labels) > 0
    assert set(labels.unique()) <= {"Bull", "Bear", "Crisis"}


def test_truncating_future_data_does_not_change_past_folds():
    """
    THE core check: refit walk-forward on the full series, then again
    on a version truncated partway through. Every fold whose test
    window ends before the truncation point must produce IDENTICAL
    predictions in both runs. If truncating the future changes a past
    fold's output, that fold was leaking future information.
    """
    master, prices, truth = test_regime.make_multi_regime_data()
    feats = build_feature_matrix(master, prices)

    full_labels, full_folds = run_walk_forward_regimes(
        feats, PRICE_TICKERS["equity"], min_train_size=300, test_size=63
    )

    # Truncate to roughly 70% of the way through and rerun.
    cutoff = feats.index[int(len(feats) * 0.7)]
    truncated_feats = feats.loc[:cutoff]
    trunc_labels, trunc_folds = run_walk_forward_regimes(
        truncated_feats, PRICE_TICKERS["equity"], min_train_size=300, test_size=63
    )

    shared_index = trunc_labels.index  # every date that exists in the truncated run
    compare = pd.DataFrame({
        "full_run": full_labels.reindex(shared_index),
        "truncated_run": trunc_labels,
    }).dropna()

    assert len(compare) > 0, "no overlapping dates to compare -- test setup issue"
    mismatches = (compare["full_run"] != compare["truncated_run"]).sum()
    assert mismatches == 0, (
        f"{mismatches} / {len(compare)} dates changed label after truncating future data "
        f"-- this means future data was leaking into past predictions"
    )


if __name__ == "__main__":
    test_splits_never_look_ahead()
    test_rolling_mode_keeps_fixed_train_size()
    test_walk_forward_runs_on_synthetic_data()
    test_truncating_future_data_does_not_change_past_folds()
    print("All checks passed -- including the direct empirical no-lookahead check.")

    master, prices, truth = test_regime.make_multi_regime_data()
    feats = build_feature_matrix(master, prices)
    labels, folds = run_walk_forward_regimes(feats, PRICE_TICKERS["equity"], min_train_size=300, test_size=63)
    truth_aligned = truth.loc[labels.index]
    crisis_days = truth_aligned[truth_aligned == "crisis"].index
    if len(crisis_days):
        hit = (labels.loc[crisis_days] == "Crisis").mean()
        print(f"\n{len(folds)} folds. Walk-forward crisis hit rate: {hit:.0%} "
              f"(lower than Phase 3's full-fit 100% is expected -- early folds "
              f"have less training history to work with, which is the honest cost "
              f"of not cheating)")
