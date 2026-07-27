"""
Phase 3 -- Regime classifier for the Regime-Shift capstone.

Fits a 3-state Gaussian HMM to a lean, deliberately-chosen feature set
and labels the resulting states Bull / Bear / Crisis.

NOTE: fit_regime_model() here fits on the FULL feature set, same as
the guide notebook's ML primer demo -- this is Phase 3's "does this
even work" checkpoint. Phase 4 (validation.py, next) redoes this fit
inside each walk-forward fold, training-data-only -- that's the
version that actually counts for the backtest.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm

from features import zscore_expanding

# Why these 3 features, and only these 3:
#   - equity 21d realized vol  -> backward-looking stress
#   - equity 21d momentum      -> trend direction (vol alone can't tell
#                                  a Bull dip from a Bear market -- you
#                                  need direction too)
#   - India VIX level          -> forward-looking, market-implied fear;
#                                 not derivable from price history at all
#
# Deliberately NOT the other ~19 Phase 2 candidates: collinear windows
# (mom_5d/21d/63d/126d for the same asset) destabilize a diag-covariance
# Gaussian HMM's fit, and per-asset vol for bonds/gold adds dimensions
# without adding regime-discriminating signal -- market regime is
# overwhelmingly an equity-driven, VIX-confirmed phenomenon, which is
# also exactly the lean feature set the guide notebook's own HMM demo
# used (log_ret/vol_21d/mom_21d).
HMM_FEATURE_TEMPLATE = ["{equity}_vol_21d", "{equity}_mom_21d", "VIX"]

N_REGIMES = 3
REGIME_NAMES = ["Bull", "Bear", "Crisis"]  # low -> high volatility ordering


def select_hmm_columns(features, equity_ticker):
    """
    Pulls the lean 3-column subset out of the Phase 2 feature pool.
    Deliberately does NOT scale here -- how these get z-scored depends
    on the context: the Phase 3 checkpoint below scales with a global
    expanding window (fine for a single "does this work" demo fit),
    while Phase 4 (validation.py) scales per fold using training-data-
    only stats, which is what the brief explicitly requires for the
    real walk-forward result. Keeping selection and scaling separate
    means both can share this one column-selection decision.
    Returns a DataFrame with columns ["vol_21d", "mom_21d", "VIX"].
    """
    cols = [c.format(equity=equity_ticker) for c in HMM_FEATURE_TEMPLATE]
    subset = features[cols].copy()
    subset.columns = ["vol_21d", "mom_21d", "VIX"]
    return subset


def fit_regime_model(X, n_components=N_REGIMES, n_init=10, random_state=42):
    """
    X: z-scored feature DataFrame/array, no NaNs.
    Fits on the FULL series -- see module docstring for why this is
    only the Phase 3 checkpoint, not the final walk-forward version.

    Gaussian HMM fitting is EM under the hood, which is only guaranteed
    to find *a* local optimum, not *the* best one -- confirmed this
    empirically while building this: a single arbitrary random_state
    converged to a degenerate fit where two states landed on nearly
    identical means (effectively only 2 real clusters instead of 3),
    while most other seeds found a clean 3-way split. Fitting n_init
    random restarts and keeping the highest-log-likelihood model is the
    standard fix, not just a workaround for one unlucky seed.
    """
    Xv = X.values if hasattr(X, "values") else X
    best_model, best_score = None, -np.inf
    for i in range(n_init):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=200,
            random_state=random_state + i,
        )
        model.fit(Xv)
        score = model.score(Xv)
        if score > best_score:
            best_model, best_score = model, score
    return best_model


def label_states(model, X, vol_col="vol_21d"):
    """
    HMM state indices (0,1,2) are arbitrary -- map them to Bull/Bear/
    Crisis by ranking states on mean volatility (lowest -> Bull,
    highest -> Crisis), same heuristic as the guide notebook.
    Returns (raw_state_array, label_map, labeled_series).
    """
    states = model.predict(X.values if hasattr(X, "values") else X)
    tmp = pd.DataFrame({"state": states, "vol": X[vol_col].values}, index=X.index)
    ranked = tmp.groupby("state")["vol"].mean().sort_values()
    label_map = {state: name for state, name in zip(ranked.index, REGIME_NAMES)}
    labels = pd.Series(states, index=X.index).map(label_map)
    return states, label_map, labels


def transition_matrix_df(model, label_map):
    """Transition matrix with Bull/Bear/Crisis row/col labels instead
    of raw state indices, ordered Bull -> Bear -> Crisis for readability."""
    ordered_states = sorted(label_map, key=lambda s: REGIME_NAMES.index(label_map[s]))
    ordered_names = [label_map[s] for s in ordered_states]
    tm = model.transmat_[np.ix_(ordered_states, ordered_states)]
    return pd.DataFrame(tm, index=ordered_names, columns=ordered_names)


def plot_regime_overlay(prices, labels, title, save_path):
    """
    The literal "regime labels plotted on top of historical price data"
    deliverable: price line with colored bands showing which regime the
    model assigned to each stretch of days. Lives here (not main.py) so
    this one file -- with only data.py/features.py as dependencies --
    can produce everything the brief's first submission bullet asks for
    on its own, without needing validation.py/portfolio.py/backtest.py.
    """
    import matplotlib.pyplot as plt

    colors = {"Bull": "#2ecc71", "Bear": "#e67e22", "Crisis": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(13, 5))
    aligned_price = prices.loc[labels.index]
    ax.plot(aligned_price.index, aligned_price.values, color="black", lw=1, zorder=3)

    vals = labels.values
    start = 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            ax.axvspan(labels.index[start], labels.index[min(i, len(vals) - 1)],
                       color=colors[vals[start]], alpha=0.25, lw=0)
            start = i

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="upper left")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    from data import load_market_data, PRICE_TICKERS
    from features import build_feature_matrix, zscore_expanding

    master, prices = load_market_data()
    feats = build_feature_matrix(master, prices)

    equity_ticker = PRICE_TICKERS["equity"]
    raw = select_hmm_columns(feats, equity_ticker)
    X = zscore_expanding(raw).dropna()  # Phase 3 checkpoint only -- Phase 4 uses fold-local scaling instead

    model = fit_regime_model(X)
    states, label_map, labels = label_states(model, X)

    print("Transition matrix:\n", transition_matrix_df(model, label_map).round(3))
    print("\nRegime counts:\n", labels.value_counts())

    plot_regime_overlay(
        prices[equity_ticker], labels,
        "HMM-inferred regimes overlaid on price (full-sample checkpoint fit)",
        "regime_overlay_phase3_checkpoint.png",
    )
    print("\nSaved: regime_overlay_phase3_checkpoint.png")