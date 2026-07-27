"""
Phase 5b -- Backtest.

Ties the walk-forward regime labels (validation.py) to the regime-
conditional optimizer (portfolio.py), applies transaction costs on
every rebalance, and compares against static 60/40 and equal-weight
benchmarks using Sharpe / Sortino / max drawdown / Calmar / turnover.

IMPORTANT: data.py's returns are LOG returns (right for the HMM
features -- they compound additively across time). Portfolio-level
returns do NOT: a portfolio's log return is not the weighted sum of
its assets' log returns (only simple returns aggregate linearly
across assets). So this module converts to SIMPLE returns first thing
-- that conversion, not the log returns themselves, is what everything
below (weighting, mu/cov estimation, equity curves) is built on.
"""

import numpy as np
import pandas as pd

from portfolio import optimize_weights

TRANSACTION_COST_BPS = 7.5  # mid-point of the 5-10bps range the brief specifies
ESTIMATION_WINDOW = 126     # trailing window (days) for estimating mu/cov at each rebalance
MIN_HISTORY = 21            # don't optimize on fewer than ~1 month of trailing data


def log_to_simple(log_returns):
    return np.exp(log_returns) - 1


def estimate_mu_cov(simple_returns, end_idx, window=ESTIMATION_WINDOW):
    """
    Trailing-window mu/cov as of end_idx, annualized -- uses only
    returns.iloc[end_idx - window : end_idx], i.e. strictly BEFORE the
    day being traded. Feeds the optimizer at each rebalance.
    """
    window_data = simple_returns.iloc[max(0, end_idx - window):end_idx]
    mu = window_data.mean().values * 252
    cov = window_data.cov().values * 252
    return mu, cov


def run_backtest(log_returns, regime_labels, transaction_cost_bps=TRANSACTION_COST_BPS,
                  estimation_window=ESTIMATION_WINDOW, min_history=MIN_HISTORY):
    """
    log_returns:   asset LOG returns, DatetimeIndex, one column/asset
                   (data.py's master, minus the VIX column)
    regime_labels: Bull/Bear/Crisis per day. For the real result this
                   MUST be validation.py's walk-forward out-of-sample
                   labels -- passing regime.py's full-sample labels
                   here would quietly reintroduce lookahead bias into
                   the one place it matters most.

    Rebalances whenever the regime changes from the last-acted-on
    regime AND there's enough trailing history to estimate mu/cov.
    This is also the literal mechanism behind the brief's warning that
    "frequent regime flips can quietly destroy returns" -- a choppy
    regime series directly means more rebalances, means more cost.

    Returns a DataFrame: portfolio_return (net of costs), regime,
    turnover (0 on non-rebalance days).
    """
    simple_returns = log_to_simple(log_returns)
    dates = simple_returns.index.intersection(regime_labels.index)
    simple_returns = simple_returns.loc[dates]
    regime_labels = regime_labels.loc[dates]
    n_assets = simple_returns.shape[1]

    weights = np.array([1 / n_assets] * n_assets)  # equal-weight until the first real rebalance
    prev_regime = None
    records = []

    for date in dates:
        regime = regime_labels.loc[date]
        end_idx = simple_returns.index.get_loc(date)
        turnover = 0.0
        cost = 0.0

        if regime != prev_regime and end_idx >= min_history:
            mu, cov = estimate_mu_cov(simple_returns, end_idx, estimation_window)
            new_weights = optimize_weights(mu, cov, regime)
            turnover = 0.5 * np.abs(new_weights - weights).sum()
            cost = turnover * (transaction_cost_bps / 10000)
            weights = new_weights
            prev_regime = regime
        elif regime != prev_regime:
            # Regime changed but there's not enough history to act on it
            # yet. Still record the change so we don't "catch up" and
            # rebalance later purely because history finally arrived.
            prev_regime = regime

        day_return = float(np.dot(weights, simple_returns.loc[date].values)) - cost
        records.append({"date": date, "regime": regime, "portfolio_return": day_return, "turnover": turnover})

        weights = weights * (1 + simple_returns.loc[date].values)
        weights = weights / weights.sum()

    return pd.DataFrame(records).set_index("date")


def run_static_benchmark(log_returns, target_weights, transaction_cost_bps=TRANSACTION_COST_BPS):
    """
    Fixed target-weight portfolio (e.g. 60/40, equal-weight), rebalanced
    back to target every calendar month -- keeps a 'static' benchmark
    from silently drifting into something else over an 8-year backtest,
    and uses the same cost treatment as the dynamic strategy for a fair
    comparison.
    """
    simple_returns = log_to_simple(log_returns)
    weights = np.array(target_weights, dtype=float)
    last_month = None
    records = []

    for date in simple_returns.index:
        month_key = (date.year, date.month)
        turnover = 0.0
        cost = 0.0
        if month_key != last_month:
            new_weights = np.array(target_weights, dtype=float)
            turnover = 0.5 * np.abs(new_weights - weights).sum()
            cost = turnover * (transaction_cost_bps / 10000)
            weights = new_weights
            last_month = month_key

        day_return = float(np.dot(weights, simple_returns.loc[date].values)) - cost
        records.append({"date": date, "portfolio_return": day_return, "turnover": turnover})

        weights = weights * (1 + simple_returns.loc[date].values)
        weights = weights / weights.sum()

    return pd.DataFrame(records).set_index("date")


def compute_metrics(daily_returns, turnover=None, periods_per_year=252):
    """Sharpe, Sortino, max drawdown, Calmar, annualized turnover. Risk-free
    rate assumed 0 -- documented simplification, swap in a real short-rate
    series if you want it precise."""
    mean_ret = daily_returns.mean() * periods_per_year
    vol = daily_returns.std() * np.sqrt(periods_per_year)
    sharpe = mean_ret / vol if vol > 0 else np.nan

    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = mean_ret / downside_vol if downside_vol and downside_vol > 0 else np.nan

    equity_curve = (1 + daily_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = drawdown.min()
    calmar = mean_ret / abs(max_dd) if max_dd < 0 else np.nan

    metrics = {
        "Ann. Return": mean_ret, "Ann. Vol": vol, "Sharpe": sharpe,
        "Sortino": sortino, "Max Drawdown": max_dd, "Calmar": calmar,
    }
    if turnover is not None:
        n_years = len(daily_returns) / periods_per_year
        metrics["Ann. Turnover"] = turnover.sum() / n_years if n_years > 0 else np.nan
    return metrics


def compare_strategies(log_returns, walk_forward_labels, asset_order,
                        transaction_cost_bps=TRANSACTION_COST_BPS):
    """
    Runs the dynamic strategy + static 60/40 + equal-weight benchmarks
    and returns one summary DataFrame, plus the individual result
    DataFrames (needed for the equity-curve / regime-chart plots).
    asset_order must match log_returns' column order, e.g.
    [equity_ticker, bonds_ticker, gold_ticker].
    """
    dynamic = run_backtest(log_returns, walk_forward_labels, transaction_cost_bps)
    sixty_forty = run_static_benchmark(log_returns, [0.6, 0.4, 0.0], transaction_cost_bps)
    equal_weight = run_static_benchmark(log_returns, [1 / 3, 1 / 3, 1 / 3], transaction_cost_bps)

    summary = pd.DataFrame({
        "Regime-Shift (dynamic)": compute_metrics(dynamic["portfolio_return"], dynamic["turnover"]),
        "Static 60/40": compute_metrics(sixty_forty["portfolio_return"], sixty_forty["turnover"]),
        "Equal-Weight": compute_metrics(equal_weight["portfolio_return"], equal_weight["turnover"]),
    }).T

    return summary, {"dynamic": dynamic, "sixty_forty": sixty_forty, "equal_weight": equal_weight}


if __name__ == "__main__":
    from data import load_market_data, PRICE_TICKERS
    from features import build_feature_matrix
    from validation import run_walk_forward_regimes

    master, prices = load_market_data()
    feats = build_feature_matrix(master, prices)
    labels, folds = run_walk_forward_regimes(feats, PRICE_TICKERS["equity"])

    asset_cols = list(PRICE_TICKERS.values())
    summary, results = compare_strategies(master[asset_cols], labels, asset_cols)
    print(summary.round(3))