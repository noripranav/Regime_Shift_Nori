"""
Regime-Shift -- full pipeline, top to bottom:
data -> features -> regime detection -> walk-forward validation ->
portfolio optimization -> backtest -> results.

Run: python main.py

Needs real network access to Yahoo Finance (this repo's dev sandbox
did not have it -- every module was built and unit-tested against
synthetic data instead; see each test_*.py file). If yfinance errors
out here, that's a network/environment issue, not a pipeline bug --
check that query1/query2.finance.yahoo.com are reachable.
"""

import matplotlib.pyplot as plt
import numpy as np

from data import load_market_data, PRICE_TICKERS
from features import build_feature_matrix, zscore_expanding
from regime import (
    select_hmm_columns, fit_regime_model, label_states, transition_matrix_df,
    plot_regime_overlay,
)
from validation import run_walk_forward_regimes
from backtest import compare_strategies


def plot_equity_curves(results, save_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in results.items():
        curve = (1 + res["portfolio_return"]).cumprod()
        ax.plot(curve.index, curve.values, label=name)
    ax.set_title("Equity curves -- Regime-Shift vs static benchmarks")
    ax.legend()
    ax.set_ylabel("Growth of 1")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def main():
    print("1/6  Loading market data...")
    master, prices = load_market_data()
    print(f"     {len(master)} trading days, {master.index.min().date()} to {master.index.max().date()}")

    print("2/6  Building features...")
    feats = build_feature_matrix(master, prices)
    equity_ticker = PRICE_TICKERS["equity"]

    print("3/6  Fitting Phase-3 checkpoint HMM (full-sample, sanity check only)...")
    raw_full = select_hmm_columns(feats, equity_ticker)
    X_full = zscore_expanding(raw_full).dropna()
    checkpoint_model = fit_regime_model(X_full)
    _, checkpoint_map, checkpoint_labels = label_states(checkpoint_model, X_full)
    print("     Transition matrix (full-sample checkpoint fit):")
    print(transition_matrix_df(checkpoint_model, checkpoint_map).round(3))
    plot_regime_overlay(
        prices[equity_ticker], checkpoint_labels,
        "Phase 3 checkpoint: HMM regimes on full-sample fit (NOT the backtest input)",
        "regime_overlay_phase3_checkpoint.png",
    )

    print("4/6  Running walk-forward regime detection (this is what the backtest actually uses)...")
    wf_labels, folds = run_walk_forward_regimes(feats, equity_ticker)
    print(f"     {len(folds)} folds, {len(wf_labels)} out-of-sample regime days")
    plot_regime_overlay(
        prices[equity_ticker], wf_labels,
        "Walk-forward (out-of-sample) HMM regimes -- the version the backtest uses",
        "regime_overlay_walk_forward.png",
    )

    print("5/6  Backtesting: dynamic strategy vs static 60/40 vs equal-weight...")
    asset_cols = list(PRICE_TICKERS.values())
    summary, results = compare_strategies(master[asset_cols], wf_labels, asset_cols)
    plot_equity_curves(results, "equity_curves.png")

    print("6/6  Done. Performance summary:\n")
    print(summary.round(3).to_string())
    summary.round(4).to_csv("results_summary.csv")
    print("\nSaved: regime_overlay_phase3_checkpoint.png, regime_overlay_walk_forward.png, "
          "equity_curves.png, results_summary.csv")


if __name__ == "__main__":
    main()