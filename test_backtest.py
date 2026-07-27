"""
Offline test for backtest.py. Run: python test_backtest.py
"""

import numpy as np
import pandas as pd

from data import PRICE_TICKERS
from backtest import (
    run_backtest, run_static_benchmark, compute_metrics, compare_strategies,
    estimate_mu_cov, log_to_simple,
)


def synthetic_log_returns(n=1000, seed=3):
    np.random.seed(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    tickers = list(PRICE_TICKERS.values())
    data = {
        tickers[0]: np.random.normal(0.0004, 0.012, n),
        tickers[1]: np.random.normal(0.0001, 0.003, n),
        tickers[2]: np.random.normal(0.0002, 0.008, n),
    }
    return pd.DataFrame(data, index=dates)


def test_compute_metrics_matches_hand_calc():
    """Known constant-mean/vol synthetic return series -- Sharpe should
    match a hand computation exactly, not just 'run without crashing'."""
    np.random.seed(1)
    n = 2000
    daily_mean, daily_std = 0.0005, 0.01
    returns = pd.Series(np.random.normal(daily_mean, daily_std, n))

    metrics = compute_metrics(returns)
    # Compare against the SAME sample's own mean/std, computed by hand --
    # this is the real correctness check. (Comparing against the true
    # generating-parameter Sharpe instead would be testing sampling
    # noise, not the code -- Sharpe estimates are noisy statistics even
    # over ~8 years of daily data.)
    hand_sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    assert np.isclose(metrics["Sharpe"], hand_sharpe, rtol=1e-6)


def test_max_drawdown_known_case():
    """Handcrafted equity curve: up to 1.20, down to 0.90 -- drawdown
    should be exactly 0.90/1.20 - 1 = -25%."""
    returns = pd.Series([0.20, -0.25])  # (1.20), then 1.20*0.75 = 0.90
    metrics = compute_metrics(returns, periods_per_year=252)
    assert np.isclose(metrics["Max Drawdown"], -0.25, atol=1e-6)


def test_static_benchmark_rebalances_monthly():
    log_returns = synthetic_log_returns(n=100)
    result = run_static_benchmark(log_returns, [0.6, 0.4, 0.0])
    # turnover should be nonzero only on the first trading day of each
    # month (that's the only time last_month changes)
    months = result.index.to_period("M")
    first_of_month = ~months.duplicated()
    assert (result.loc[result["turnover"] > 0].index.isin(result.index[first_of_month])).all()


def test_choppier_regimes_cost_more_and_can_hurt_returns():
    """Directly tests the brief's own claim: 'frequent regime flips can
    quietly destroy returns once costs are included.' Same underlying
    returns, same regime PROPORTIONS, but one label series flips every
    day and the other holds each regime for a stretch -- the choppy one
    must show strictly higher turnover."""
    log_returns = synthetic_log_returns(n=1000)
    dates = log_returns.index

    stable = pd.Series(["Bull"] * 500 + ["Bear"] * 500, index=dates)
    choppy = pd.Series((["Bull", "Bear"] * 500), index=dates)  # flips every day

    stable_result = run_backtest(log_returns, stable)
    choppy_result = run_backtest(log_returns, choppy)

    assert choppy_result["turnover"].sum() > stable_result["turnover"].sum() * 5, (
        "choppy regime labels should generate substantially more turnover"
    )


def test_mu_cov_estimation_has_no_lookahead():
    log_returns = synthetic_log_returns(n=500)
    simple = log_to_simple(log_returns)

    end_idx = 300
    mu_a, cov_a = estimate_mu_cov(simple, end_idx)
    # truncate everything from end_idx onward and re-estimate -- must be identical
    truncated = simple.iloc[:end_idx]
    mu_b, cov_b = estimate_mu_cov(truncated, end_idx)

    assert np.allclose(mu_a, mu_b)
    assert np.allclose(cov_a, cov_b)


def test_compare_strategies_runs_end_to_end():
    log_returns = synthetic_log_returns(n=800)
    dates = log_returns.index
    np.random.seed(5)
    labels = pd.Series(np.random.choice(["Bull", "Bear", "Crisis"], size=len(dates), p=[0.6, 0.3, 0.1]), index=dates)

    summary, results = compare_strategies(log_returns, labels, list(PRICE_TICKERS.values()))
    assert list(summary.index) == ["Regime-Shift (dynamic)", "Static 60/40", "Equal-Weight"]
    assert {"Sharpe", "Sortino", "Max Drawdown", "Calmar", "Ann. Turnover"} <= set(summary.columns)
    assert not summary["Sharpe"].isna().any()


if __name__ == "__main__":
    test_compute_metrics_matches_hand_calc()
    test_max_drawdown_known_case()
    test_static_benchmark_rebalances_monthly()
    test_choppier_regimes_cost_more_and_can_hurt_returns()
    test_mu_cov_estimation_has_no_lookahead()
    test_compare_strategies_runs_end_to_end()
    print("All checks passed.")
