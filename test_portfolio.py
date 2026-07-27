"""
Offline test for portfolio.py. No network/data dependency -- just
checks the optimizer respects its constraints and produces weights
that move in the economically sensible direction across regimes.

Run: python test_portfolio.py
"""

import numpy as np

from portfolio import optimize_weights


def toy_mu_cov():
    mu = np.array([0.12, 0.05, 0.07])
    vols = np.array([0.18, 0.06, 0.14])
    corr = np.array([
        [1.00, -0.15, 0.05],
        [-0.15, 1.00, 0.10],
        [0.05, 0.10, 1.00],
    ])
    cov = np.outer(vols, vols) * corr
    return mu, cov


def test_weights_are_valid_portfolio():
    mu, cov = toy_mu_cov()
    for regime in ["Bull", "Bear", "Crisis"]:
        w = optimize_weights(mu, cov, regime)
        assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{regime} weights don't sum to 1: {w.sum()}"
        assert (w >= -1e-8).all(), f"{regime} produced a negative weight: {w}"


def test_equity_allocation_shrinks_bull_to_crisis():
    """Equity is the highest-return, highest-vol asset -- allocation to
    it should monotonically shrink as the regime gets more defensive."""
    mu, cov = toy_mu_cov()
    w_bull = optimize_weights(mu, cov, "Bull")
    w_bear = optimize_weights(mu, cov, "Bear")
    w_crisis = optimize_weights(mu, cov, "Crisis")
    assert w_bull[0] > w_bear[0] > w_crisis[0], (
        f"expected equity weight Bull > Bear > Crisis, got "
        f"{w_bull[0]:.2%} / {w_bear[0]:.2%} / {w_crisis[0]:.2%}"
    )


def test_crisis_ignores_expected_return():
    """Crisis is pure min-variance -- changing expected returns shouldn't
    move the Crisis weights at all (only the covariance matters)."""
    _, cov = toy_mu_cov()
    mu_a = np.array([0.12, 0.05, 0.07])
    mu_b = np.array([0.30, -0.10, 0.20])  # wildly different expected returns
    w_a = optimize_weights(mu_a, cov, "Crisis")
    w_b = optimize_weights(mu_b, cov, "Crisis")
    assert np.allclose(w_a, w_b, atol=1e-4), "Crisis weights should be invariant to mu"


def test_unknown_regime_raises():
    mu, cov = toy_mu_cov()
    try:
        optimize_weights(mu, cov, "Sideways")
        assert False, "should have raised on an unrecognized regime label"
    except ValueError:
        pass


if __name__ == "__main__":
    test_weights_are_valid_portfolio()
    test_equity_allocation_shrinks_bull_to_crisis()
    test_crisis_ignores_expected_return()
    test_unknown_regime_raises()
    print("All checks passed.")
