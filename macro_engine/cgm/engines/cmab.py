"""
engines/cmab.py
Contextual Multi-Armed Bandit (CMAB) engine.

Implements the UCB and Thompson Sampling algorithms used in Chapters 2–3 of
Computational Global Macro to simulate iterated Bayesian games.

A "context" is a vector of (payoff, probability) pairs — one per arm — that
represents the type-state the game is in for a given round.  Multiple contexts
are sampled according to their prior probabilities at each time-step, exactly
as described in the book's Run-Pass and Colonel-Blotto examples.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    """One action available to an agent."""
    name: str
    base_payoff: float          # deterministic payoff when this arm "succeeds"
    success_prob: float = 1.0   # probability the payoff materialises


@dataclass
class Context:
    """
    A type-state (Bayesian game context).

    arms        : list of Arm objects for this context
    prior       : probability this context is drawn at any round
    label       : human-readable name (e.g. "Strong Passer")
    """
    arms: list[Arm]
    prior: float
    label: str = ""

    def expected_payoff(self, arm_idx: int) -> float:
        a = self.arms[arm_idx]
        return a.base_payoff * a.success_prob


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

class UCBAgent:
    """
    Upper-Confidence-Bound algorithm (Chapter 2, Table 2.9).

        a* = argmax [ Q(a) + c * sqrt( log(t) / N(a) ) ]

    Parameters
    ----------
    n_arms  : number of actions
    c       : exploration constant (higher = more exploration)
    """

    def __init__(self, n_arms: int, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c
        self.Q = np.zeros(n_arms)      # estimated mean payoffs
        self.N = np.zeros(n_arms)      # pull counts
        self.t = 0
        self.history: list[dict] = []

    def select(self) -> int:
        self.t += 1
        # Initialise each arm once
        untried = np.where(self.N == 0)[0]
        if len(untried):
            return int(untried[0])
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return int(np.argmax(ucb))

    def update(self, arm: int, reward: float):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]
        self.history.append({"t": self.t, "arm": arm, "reward": reward,
                              "Q": self.Q.copy()})

    def reset(self):
        self.Q[:] = 0; self.N[:] = 0; self.t = 0; self.history.clear()


class ThompsonAgent:
    """
    Thompson Sampling algorithm (Chapter 5, Table 5.3).

    Maintains Beta(alpha_k, beta_k) posteriors over each arm's success rate.
    At each step samples theta_k ~ Beta and pulls the arm with highest sample.

    Parameters
    ----------
    n_arms          : number of arms
    alpha0, beta0   : symmetric Beta prior (default: uniform)
    """

    def __init__(self, n_arms: int, alpha0: float = 1.0, beta0: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.full(n_arms, alpha0, dtype=float)
        self.beta_ = np.full(n_arms, beta0, dtype=float)
        self.history: list[dict] = []
        self.t = 0

    def select(self) -> int:
        samples = np.random.beta(self.alpha, self.beta_)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float, threshold: float = 0.0):
        """reward > threshold counts as a "success" for the Beta update."""
        self.t += 1
        if reward > threshold:
            self.alpha[arm] += 1
        else:
            self.beta_[arm] += 1
        self.history.append({"t": self.t, "arm": arm, "reward": reward,
                              "alpha": self.alpha.copy(),
                              "beta": self.beta_.copy()})

    def reset(self):
        n = self.n_arms
        self.alpha[:] = 1; self.beta_[:] = 1
        self.history.clear(); self.t = 0


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

class CMABSimulation:
    """
    Runs an iterated Bayesian game using a CMAB agent.

    At each round:
      1. Sample a context according to context priors.
      2. Agent selects an arm.
      3. Realise reward from the arm's (payoff, success_prob) in that context.
      4. Agent updates its estimates.

    Parameters
    ----------
    contexts    : list of Context objects (types)
    agent       : UCBAgent or ThompsonAgent
    n_rounds    : rounds per simulation
    n_iters     : Monte-Carlo iterations (results are averaged)
    rng_seed    : reproducibility
    """

    def __init__(self,
                 contexts: list[Context],
                 agent: UCBAgent | ThompsonAgent,
                 n_rounds: int = 1000,
                 n_iters: int = 500,
                 rng_seed: int | None = 42):
        self.contexts = contexts
        self.agent = agent
        self.n_rounds = n_rounds
        self.n_iters = n_iters
        self.rng = np.random.default_rng(rng_seed)

        priors = np.array([c.prior for c in contexts])
        assert abs(priors.sum() - 1.0) < 1e-6, "Context priors must sum to 1."
        self.priors = priors

        n_arms = len(contexts[0].arms)
        assert all(len(c.arms) == n_arms for c in contexts), \
            "All contexts must have the same number of arms."
        self.n_arms = n_arms

    def _run_once(self) -> np.ndarray:
        """Single Monte-Carlo iteration. Returns (n_rounds, n_arms) payoff array."""
        self.agent.reset()
        payoffs = np.zeros((self.n_rounds, self.n_arms))
        for r in range(self.n_rounds):
            ctx_idx = self.rng.choice(len(self.contexts), p=self.priors)
            ctx = self.contexts[ctx_idx]
            arm = self.agent.select()
            # Stochastic reward
            a = ctx.arms[arm]
            reward = a.base_payoff if self.rng.random() < a.success_prob else 0.0
            self.agent.update(arm, reward)
            # Record expected payoffs (not just realised) for stable plots
            payoffs[r] = [ctx.expected_payoff(i) * (self.agent.N[i] > 0 or 1)
                          for i in range(self.n_arms)]
        return payoffs

    def run(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          avg_payoffs   : (n_rounds, n_arms) mean payoff trajectories
          cum_payoffs   : (n_rounds, n_arms) cumulative mean payoffs
          arm_names     : list of arm name strings
          best_arm      : index of arm with highest long-run average payoff
        """
        all_payoffs = np.stack([self._run_once() for _ in range(self.n_iters)])
        avg = all_payoffs.mean(axis=0)           # (n_rounds, n_arms)
        cum = np.cumsum(avg, axis=0) / (np.arange(self.n_rounds)[:, None] + 1)
        arm_names = [self.contexts[0].arms[i].name for i in range(self.n_arms)]
        best_arm = int(np.argmax(cum[-1]))
        return {
            "avg_payoffs": avg,
            "cum_payoffs": cum,
            "arm_names": arm_names,
            "best_arm": best_arm,
            "best_arm_name": arm_names[best_arm],
        }