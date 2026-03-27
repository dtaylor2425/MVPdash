"""
portfolio/optimizer.py
Portfolio construction layer (Chapter 6).

Implements:
  1. Black-Litterman (BL) — Bayesian view-blending into posterior returns
  2. Ordinal Black-Litterman (OBL) — ranked views instead of precise forecasts
  3. Mean-Variance Optimization (MVO) on any return vector
  4. Robust MVO with estimation-error shrinkage (Fabozzi et al. 2007)
  5. GAN-based scenario stress testing placeholder (full GAN = separate module)

All optimizers accept MacroView objects from engines/causal.py as input,
making the game theory → causal → portfolio pipeline seamless.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize, linprog
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Helper: build BL P and Q matrices from MacroViews
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    method: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        w_str = ", ".join(f"{w:.1%}" for w in self.weights)
        return (f"PortfolioResult({self.method})\n"
                f"  weights    : [{w_str}]\n"
                f"  exp return : {self.expected_return:.2%}\n"
                f"  volatility : {self.volatility:.2%}\n"
                f"  Sharpe     : {self.sharpe:.2f}")


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

class BlackLitterman:
    """
    Standard Black-Litterman model (Chapter 6, Section 6.3.2).

    Parameters
    ----------
    assets          : list of asset names
    sigma           : (n,n) covariance matrix of asset returns
    w_market        : (n,) market-cap benchmark weights
    risk_aversion   : γ parameter (typically 2.5–3.5)
    tau             : uncertainty scalar on prior (typically 0.01–0.05)
    """

    def __init__(self,
                 assets: list[str],
                 sigma: np.ndarray,
                 w_market: np.ndarray,
                 risk_aversion: float = 2.5,
                 tau: float = 0.025):
        self.assets = list(assets)
        self.n = len(assets)
        self.sigma = np.array(sigma)
        self.w_market = np.array(w_market)
        self.risk_aversion = risk_aversion
        self.tau = tau

        assert self.sigma.shape == (self.n, self.n)
        assert len(self.w_market) == self.n

        # Implied equilibrium excess returns (reverse-optimization)
        self.pi = risk_aversion * sigma @ w_market

    def _build_view_matrices(self, views) -> tuple[np.ndarray, np.ndarray,
                                                    np.ndarray]:
        """
        Build P (k×n picking matrix), Q (k,) return vector, Omega (k×k
        diagonal uncertainty matrix) from a list of MacroView objects.
        """
        try:
            from cgm.engines.causal import MacroView
        except ImportError:
            from engines.causal import MacroView
        k = len(views)
        P = np.zeros((k, self.n))
        Q = np.zeros(k)
        omega_diag = np.zeros(k)

        for i, v in enumerate(views):
            if v.asset not in self.assets:
                raise ValueError(f"Asset '{v.asset}' not in portfolio.")
            j = self.assets.index(v.asset)
            P[i, j] = 1.0
            Q[i] = v.view_return()
            # Omega_ii = tau * P_i * Sigma * P_i' scaled by view uncertainty
            omega_diag[i] = (self.tau * P[i] @ self.sigma @ P[i] /
                             max(v.confidence, 1e-6))

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def posterior(self, views) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute BL posterior mean and covariance.

        Returns
        -------
        mu_bl   : (n,) posterior expected excess returns
        sigma_bl: (n,n) posterior covariance
        """
        P, Q, Omega = self._build_view_matrices(views)
        tau_sig = self.tau * self.sigma
        # BL formula: μ̃ = π + τΣP'[τPΣP' + Ω]⁻¹ [Q - Pπ]
        M = P @ tau_sig @ P.T + Omega
        K = tau_sig @ P.T @ np.linalg.inv(M)
        mu_bl = self.pi + K @ (Q - P @ self.pi)
        sigma_bl = self.sigma + tau_sig - K @ P @ tau_sig
        return mu_bl, sigma_bl

    def optimize(self, views,
                 w_min: float = 0.0,
                 w_max: float = 1.0) -> PortfolioResult:
        """
        Mean-variance optimize on BL posterior returns.

        Parameters
        ----------
        views   : list of MacroView
        w_min   : lower bound on each weight
        w_max   : upper bound on each weight
        """
        mu, sig = self.posterior(views)
        return _mvo(mu, sig, self.risk_aversion, self.assets,
                    w_min, w_max, method="Black-Litterman MVO")


# ---------------------------------------------------------------------------
# Ordinal Black-Litterman (Chapter 6, Section 6.3.3)
# ---------------------------------------------------------------------------

class OrdinalBL:
    """
    Ordinal Black-Litterman — accepts ranked views on returns, variances,
    and covariances rather than precise numeric forecasts.

    Parameters
    ----------
    assets          : list of asset names
    hist_returns    : (n,) historical mean returns
    sigma           : (n,n) historical covariance matrix
    w_market        : (n,) benchmark weights
    tau             : uncertainty scalar
    """

    def __init__(self,
                 assets: list[str],
                 hist_returns: np.ndarray,
                 sigma: np.ndarray,
                 w_market: np.ndarray,
                 tau: float = 0.025):
        self.assets = list(assets)
        self.n = len(assets)
        self.h = np.array(hist_returns)
        self.sigma = np.array(sigma)
        self.w = np.array(w_market)
        self.tau = tau
        self.E_prior = self.w * self.h   # prior expected returns

    def optimize(self,
                 return_ranks: np.ndarray,
                 variance_ranks: np.ndarray,
                 covariance_ranks: np.ndarray | None = None,
                 w_min: float = -0.5,
                 w_max: float = 0.5) -> PortfolioResult:
        """
        Compute OBL posterior and optimize.

        Parameters
        ----------
        return_ranks    : (n,) higher = prefer higher expected return
        variance_ranks  : (n,) higher = prefer lower variance
        covariance_ranks: (n,) higher = prefer lower covariance with others
                          If None, skipped.
        """
        n, tau = self.n, self.tau
        sig = self.sigma
        I = np.eye(n)

        # Posterior variance (eq. 6.6)
        Q_var = variance_ranks / variance_ranks.sum()
        V_post = sig + tau * (sig - tau * sig @ np.outer(Q_var, Q_var) @ sig)

        # Posterior expected returns (eq. 6.7)
        Q_ret = return_ranks / return_ranks.sum()
        E_post = self.E_prior + tau * sig @ (Q_ret - self.E_prior)

        # Posterior covariance adjustment (eq. 6.8)
        if covariance_ranks is not None:
            Q_cov = covariance_ranks / covariance_ranks.sum()
            C_post = sig + tau * sig @ (Q_cov - sig @ Q_cov)
        else:
            C_post = V_post

        return _mvo(E_post, C_post, 2.5, self.assets, w_min, w_max,
                    method="Ordinal Black-Litterman")


# ---------------------------------------------------------------------------
# Robust MVO with estimation error shrinkage (Chapter 6, Section 6.4)
# ---------------------------------------------------------------------------

class RobustMVO:
    """
    Robust Mean-Variance Optimization.

    Adds a shrinkage penalty on assets with high return estimation uncertainty,
    implementing the spirit of Fabozzi et al. (2007) and equation (6.9):

        max  E(r)'w - delta'*psi - (gamma/2) w'Σw
        s.t. sum(w) = 1, w >= w_min, w <= w_max

    Parameters
    ----------
    assets          : asset names
    sigma           : (n,n) covariance matrix
    risk_aversion   : γ
    estimation_error: (n,) vector — uncertainty in each return estimate.
                      Higher = more shrinkage toward zero weight.
                      Typically set to std-error of mean return estimate.
    """

    def __init__(self,
                 assets: list[str],
                 sigma: np.ndarray,
                 risk_aversion: float = 2.5,
                 estimation_error: np.ndarray | None = None):
        self.assets = list(assets)
        self.n = len(assets)
        self.sigma = np.array(sigma)
        self.risk_aversion = risk_aversion
        self.delta = (estimation_error if estimation_error is not None
                      else np.zeros(self.n))

    def optimize(self, mu: np.ndarray,
                 w_min: float = 0.0,
                 w_max: float = 1.0) -> PortfolioResult:
        """
        Parameters
        ----------
        mu      : (n,) expected excess returns (e.g. from BL posterior)
        """
        mu = np.array(mu)
        delta = self.delta
        gamma = self.risk_aversion
        sig = self.sigma

        def neg_utility(w):
            return -(mu @ w - delta @ np.abs(w) -
                     0.5 * gamma * w @ sig @ w)

        def neg_utility_grad(w):
            return -(mu - delta * np.sign(w) - gamma * sig @ w)

        n = self.n
        w0 = np.full(n, 1.0 / n)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(w_min, w_max)] * n

        res = minimize(neg_utility, w0, jac=neg_utility_grad,
                       method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 1000})

        w = res.x
        ret = float(mu @ w)
        vol = float(np.sqrt(w @ sig @ w))
        sharpe = ret / vol if vol > 0 else 0.0
        return PortfolioResult(
            weights=w, expected_return=ret, volatility=vol,
            sharpe=sharpe, method="Robust MVO",
            metadata={"estimation_error": delta.tolist(),
                      "converged": res.success})


# ---------------------------------------------------------------------------
# Linear programming for causal views (Chapter 6, Section 6.2)
# ---------------------------------------------------------------------------

def causal_lp_trades(
    causal_distances: dict[str, float],
    buy_assets: list[str],
    sell_assets: list[str],
    total_budget: float = 0.05,
    min_trade: float = 0.005,
    max_trade: float = 0.05,
) -> dict[str, float]:
    """
    Linear programme that minimises total causal distance-weighted exposure.
    Implements equation (6.2) from the book.

    Parameters
    ----------
    causal_distances: {scenario_label: distance_score} from causal graph analysis
    buy_assets      : assets to buy (one per scenario)
    sell_assets     : assets to sell (one per scenario)
    total_budget    : sum of |trades| = total_budget (dollar-neutral)
    min/max_trade   : per-leg size bounds

    Returns
    -------
    dict mapping each asset to its trade size (+= buy, -= sell)
    """
    scenarios = sorted(causal_distances.keys())
    n = len(scenarios)
    assert len(buy_assets) == n and len(sell_assets) == n

    dist = np.array([causal_distances[s] for s in scenarios])

    # Variables: x[0..n-1] = buy sizes, x[n..2n-1] = sell sizes (positive)
    # Objective: minimise sum(dist_i * buy_i) - sum(dist_i * sell_i)
    c = np.concatenate([dist, -dist])

    # Equality: buys + sells = 0 net → sum(buys) - sum(sells) = 0
    A_eq = np.concatenate([np.ones(n), -np.ones(n)])[None, :]
    b_eq = np.array([0.0])

    # Budget constraint: sum(buys) in [total_budget - tol, total_budget + tol]
    tol = 0.005
    A_ub = np.vstack([
        np.concatenate([np.ones(n), np.zeros(n)]),    # sum buys <= budget+tol
        -np.concatenate([np.ones(n), np.zeros(n)]),   # sum buys >= budget-tol
    ])
    b_ub = np.array([total_budget + tol, -(total_budget - tol)])

    bounds = [(min_trade, max_trade)] * (2 * n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        return {}

    trades = {}
    for i, asset in enumerate(buy_assets):
        trades[asset] = trades.get(asset, 0.0) + res.x[i]
    for i, asset in enumerate(sell_assets):
        trades[asset] = trades.get(asset, 0.0) - res.x[n + i]
    return trades


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mvo(mu, sigma, gamma, assets, w_min, w_max, method) -> PortfolioResult:
    """Shared mean-variance optimization core."""
    n = len(mu)

    def neg_utility(w):
        return -(mu @ w - 0.5 * gamma * w @ sigma @ w)

    def neg_utility_grad(w):
        return -(mu - gamma * sigma @ w)

    w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(w_min, w_max)] * n

    res = minimize(neg_utility, w0, jac=neg_utility_grad,
                   method="SLSQP", bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 500})

    w = res.x
    ret = float(mu @ w)
    vol = float(np.sqrt(w @ sigma @ w))
    sharpe = ret / vol if vol > 0 else 0.0
    return PortfolioResult(weights=w, expected_return=ret,
                           volatility=vol, sharpe=sharpe, method=method,
                           metadata={"converged": res.success})