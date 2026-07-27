"""
Phase 5a -- Regime-conditional portfolio optimization.

Solves for optimal weights (equity/bonds/gold) under a convex objective
that depends on the CURRENT regime: Bull leans into risk-adjusted
return, Bear turns more defensive, Crisis drops straight to
minimum-variance. All three are long-only, fully-invested QPs, solved
with cvxpy.
"""

import numpy as np
import cvxpy as cp

# Risk aversion by regime -- higher = more conservative. Crisis doesn't
# use this at all: it's pure minimum-variance, ignoring expected return
# outright, which is the literal "minimize volatility in Crisis" the
# brief asks for.
RISK_AVERSION = {"Bull": 2.0, "Bear": 6.0}


def optimize_weights(mu, cov, regime, risk_aversion=None):
    """
    mu:     expected returns, one value per asset (e.g. trailing mean
            daily log return over some estimation window)
    cov:    covariance matrix of returns, same asset order as mu
    regime: "Bull" | "Bear" | "Crisis"

    Returns a long-only, fully-invested weight vector (sum(w)=1, w>=0)
    -- no leverage, no shorting. That's a discretionary choice (matches
    holding 3 plain ETFs, no derivatives/margin), not a project
    requirement -- worth calling out as such in the README.
    """
    risk_aversion = risk_aversion or RISK_AVERSION
    mu = np.asarray(mu)
    cov = np.asarray(cov)
    n = len(mu)

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]

    if regime == "Crisis":
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov)))
    else:
        if regime not in risk_aversion:
            raise ValueError(f"unknown regime '{regime}', expected one of "
                              f"{['Crisis'] + list(risk_aversion)}")
        lam = risk_aversion[regime]
        # Mean-variance utility: maximize return net of a risk penalty.
        # This is the convex, well-posed stand-in for "maximize Sharpe" --
        # true Sharpe (return / vol) is a ratio, not a convex QP objective
        # on its own. A lower lambda in calmer regimes has the same
        # practical effect (leans harder into the return term).
        objective = cp.Maximize(mu @ w - lam * cp.quad_form(w, cp.psd_wrap(cov)))

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"optimization did not converge cleanly: {problem.status}")

    # Numerical cleanup: cvxpy can return e.g. -1e-11 for a weight that's
    # mathematically exactly 0 under the w>=0 constraint.
    weights = np.clip(w.value, 0, None)
    return weights / weights.sum()


if __name__ == "__main__":
    # Toy 3-asset example: equity high return/high vol, bonds low
    # return/low vol, gold moderate/moderate, with realistic-ish
    # correlations (equity-bonds slightly negative, equity-gold ~0).
    mu = np.array([0.12, 0.05, 0.07])  # annualized, illustrative
    vols = np.array([0.18, 0.06, 0.14])
    corr = np.array([
        [1.00, -0.15, 0.05],
        [-0.15, 1.00, 0.10],
        [0.05, 0.10, 1.00],
    ])
    cov = np.outer(vols, vols) * corr

    for regime in ["Bull", "Bear", "Crisis"]:
        w = optimize_weights(mu, cov, regime)
        print(f"{regime:>7}: equity={w[0]:.2%}  bonds={w[1]:.2%}  gold={w[2]:.2%}")