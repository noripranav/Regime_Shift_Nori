"""
Phase 4 -- Walk-forward validation harness.

Refits the regime HMM separately inside each walk-forward fold, using
ONLY that fold's training data to both fit the model AND compute the
z-scoring stats -- never the full dataset, never the test fold's own
stats. This is the piece that makes the regime labels legitimately
usable in a backtest instead of secretly knowing the future.

Note this uses a DIFFERENT z-scoring scheme than regime.py's Phase 3
checkpoint: regime.py's __main__ block does one continuous expanding
z-score across the whole series (fine for a quick full-fit sanity
check). Here, each fold computes its own flat mean/std from ONLY that
fold's training slice and applies it to both train and test -- the
exact sklearn scaler.fit(train)/transform(test) analogue the guide
notebook's Section 8 describes as "the correct loop".
"""

import numpy as np
import pandas as pd

from regime import select_hmm_columns, fit_regime_model, label_states, N_REGIMES
from features import CLIP_SIGMA


def walk_forward_splits(n_obs, min_train_size=504, test_size=63, mode="expanding"):
    """
    Yields (train_idx, test_idx) arrays that slide forward in time.
      mode="expanding": training window always starts at 0, grows each fold.
      mode="rolling":    training window is a fixed size (min_train_size),
                         slides forward instead of growing.
    Defaults: ~2yr initial training window, ~1qtr test windows.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown mode: {mode}")

    splits = []
    start_test = min_train_size
    while start_test < n_obs:
        end_test = min(start_test + test_size, n_obs)
        if mode == "expanding":
            train_idx = np.arange(0, start_test)
        else:
            train_idx = np.arange(max(0, start_test - min_train_size), start_test)
        test_idx = np.arange(start_test, end_test)
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
        start_test += test_size
    return splits


def run_walk_forward_regimes(features, equity_ticker, min_train_size=504,
                              test_size=63, mode="expanding", n_init=10,
                              random_state=42):
    """
    The core walk-forward loop. For every fold:
      1. z-score using ONLY that fold's training slice mean/std
      2. fit a fresh HMM on the scaled training slice only
      3. predict regimes on the scaled test slice with that fold's model
      4. keep ONLY the test-fold predictions (out-of-sample)

    Concatenating every fold's out-of-sample predictions gives one
    regime label per day that a live system could actually have
    produced at the time. This -- not the Phase 3 full-sample fit --
    is the series the backtest (Phase 5) consumes.

    Returns (walk_forward_labels, fold_info) where fold_info is a list
    of dicts with each fold's date ranges and label_map, useful for
    debugging / the README's "how it was validated" section.
    """
    raw = select_hmm_columns(features, equity_ticker)
    splits = walk_forward_splits(len(raw), min_train_size, test_size, mode)
    if not splits:
        raise ValueError("no folds produced -- min_train_size is probably >= len(features)")

    all_labels = []
    fold_info = []
    for train_idx, test_idx in splits:
        train_data = raw.iloc[train_idx]
        test_data = raw.iloc[test_idx]

        mu, sigma = train_data.mean(), train_data.std()
        sigma = sigma.where(sigma > 1e-10)  # ~0 std -> NaN, not inf (see features.zscore_expanding)
        train_scaled = ((train_data - mu) / sigma).clip(-CLIP_SIGMA, CLIP_SIGMA)
        test_scaled = ((test_data - mu) / sigma).clip(-CLIP_SIGMA, CLIP_SIGMA)

        if train_scaled.isna().any().any():
            flat_cols = train_scaled.columns[train_scaled.isna().any()].tolist()
            raise ValueError(
                f"Fold {raw.index[train_idx[0]].date()} to {raw.index[train_idx[-1]].date()}: "
                f"{flat_cols} had ~zero variance across this whole training window (a stale/flat "
                f"price stretch, most likely) -- z-scoring it is meaningless. This is a real data "
                f"quality issue to look at, not something to silently paper over."
            )

        model = fit_regime_model(train_scaled, n_components=N_REGIMES,
                                  n_init=n_init, random_state=random_state)
        _, label_map, _ = label_states(model, train_scaled)

        test_states = model.predict(test_scaled.values)
        test_labels = pd.Series(test_states, index=test_scaled.index).map(label_map)

        all_labels.append(test_labels)
        fold_info.append({
            "train_range": (raw.index[train_idx[0]], raw.index[train_idx[-1]]),
            "test_range": (raw.index[test_idx[0]], raw.index[test_idx[-1]]),
            "label_map": label_map,
        })

    walk_forward_labels = pd.concat(all_labels).sort_index()
    return walk_forward_labels, fold_info


if __name__ == "__main__":
    from data import load_market_data, PRICE_TICKERS
    from features import build_feature_matrix

    master, prices = load_market_data()
    feats = build_feature_matrix(master, prices)
    labels, folds = run_walk_forward_regimes(feats, PRICE_TICKERS["equity"])

    print(f"{len(folds)} folds, {len(labels)} out-of-sample regime days total")
    print(labels.value_counts())