"""
engines/qlearning.py
Q-learning engine for multi-state geopolitical games (Chapter 4).

Supports:
  - Standard Q-learning (off-policy, state-action value estimation)
  - SARSA (on-policy variant used in Chapter 5's trade negotiation game)
  - Two-player games where each agent may use a different algorithm

Key difference from CMAB: agents have a STATE that transitions based on
actions, enabling War of Attrition, Stackelberg, Signaling, and
Pursuit-Evasion games.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Game specification
# ---------------------------------------------------------------------------

@dataclass
class GameSpec:
    """
    Defines a two-player, finite-state, finite-action game.

    n_states        : number of world states
    n_actions_p1    : action space size for Player 1
    n_actions_p2    : action space size for Player 2
    payoff_fn       : (state, a1, a2, rng) -> (reward_p1, reward_p2, next_state)
    action_names_p1 : optional labels for Player 1's actions
    action_names_p2 : optional labels for Player 2's actions
    state_names     : optional labels for states
    """
    n_states: int
    n_actions_p1: int
    n_actions_p2: int
    payoff_fn: Callable[[int, int, int, np.random.Generator],
                        tuple[float, float, int]]
    action_names_p1: list[str] = field(default_factory=list)
    action_names_p2: list[str] = field(default_factory=list)
    state_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Off-policy Q-learning (Chapter 4).

        Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]

    Parameters
    ----------
    n_states    : number of world states
    n_actions   : number of actions available
    alpha       : learning rate
    gamma       : discount factor
    epsilon     : epsilon-greedy exploration probability
    """

    algo = "Q-learning"

    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.history: list[dict] = []

    def select(self, state: int, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.Q[state]))

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int | None = None):
        """a_next is ignored for Q-learning (uses max); provided for SARSA compatibility."""
        td_target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
        self.history.append({"s": s, "a": a, "r": r, "s_next": s_next,
                              "Q_sa": self.Q[s, a]})

    def reset(self):
        self.Q[:] = 0
        self.history.clear()


class SARSAAgent(QLearningAgent):
    """
    On-policy SARSA (Chapter 5).

        Q(s,a) <- Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]

    Uses the actual next action a' under current policy — unlike Q-learning
    which uses the greedy max.  More conservative in risky environments.
    """

    algo = "SARSA"

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int | None = None):
        if a_next is None:
            raise ValueError("SARSA requires a_next.")
        td_target = r + self.gamma * self.Q[s_next, a_next]
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])
        self.history.append({"s": s, "a": a, "r": r, "s_next": s_next,
                              "a_next": a_next, "Q_sa": self.Q[s, a]})


# ---------------------------------------------------------------------------
# Two-player simulation runner
# ---------------------------------------------------------------------------

class TwoPlayerSimulation:
    """
    Runs a two-player game where each player may use a different algorithm.

    Parameters
    ----------
    spec        : GameSpec describing payoffs and transitions
    agent1      : QLearningAgent or SARSAAgent for Player 1
    agent2      : QLearningAgent or SARSAAgent for Player 2
    n_rounds    : rounds per simulation episode
    n_iters     : Monte-Carlo iterations
    init_state  : starting state (default 0)
    rng_seed    : reproducibility
    """

    def __init__(self,
                 spec: GameSpec,
                 agent1: QLearningAgent,
                 agent2: QLearningAgent,
                 n_rounds: int = 50,
                 n_iters: int = 1000,
                 init_state: int = 0,
                 rng_seed: int | None = 42):
        self.spec = spec
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_rounds = n_rounds
        self.n_iters = n_iters
        self.init_state = init_state
        self.rng = np.random.default_rng(rng_seed)

    def _run_once(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rewards_p1, rewards_p2) arrays of length n_rounds."""
        self.agent1.reset()
        self.agent2.reset()
        s = self.init_state
        r1_hist = np.zeros(self.n_rounds)
        r2_hist = np.zeros(self.n_rounds)

        a1 = self.agent1.select(s, self.rng)
        a2 = self.agent2.select(s, self.rng)

        for t in range(self.n_rounds):
            r1, r2, s_next = self.spec.payoff_fn(s, a1, a2, self.rng)
            r1_hist[t] = r1
            r2_hist[t] = r2

            a1_next = self.agent1.select(s_next, self.rng)
            a2_next = self.agent2.select(s_next, self.rng)

            self.agent1.update(s, a1, r1, s_next, a1_next)
            self.agent2.update(s, a2, r2, s_next, a2_next)

            s, a1, a2 = s_next, a1_next, a2_next

        return r1_hist, r2_hist

    def run(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          avg_rewards_p1, avg_rewards_p2  : (n_rounds,) mean reward trajectories
          cum_avg_p1, cum_avg_p2          : cumulative averages
          final_Q1, final_Q2              : Q-tables after last iteration
          algo_p1, algo_p2               : algorithm names
        """
        all_r1, all_r2 = [], []
        for _ in range(self.n_iters):
            r1, r2 = self._run_once()
            all_r1.append(r1)
            all_r2.append(r2)

        avg1 = np.mean(all_r1, axis=0)
        avg2 = np.mean(all_r2, axis=0)
        t = np.arange(1, self.n_rounds + 1)
        return {
            "avg_rewards_p1": avg1,
            "avg_rewards_p2": avg2,
            "cum_avg_p1": np.cumsum(avg1) / t,
            "cum_avg_p2": np.cumsum(avg2) / t,
            "final_Q1": self.agent1.Q.copy(),
            "final_Q2": self.agent2.Q.copy(),
            "algo_p1": self.agent1.algo,
            "algo_p2": self.agent2.algo,
        }


# ---------------------------------------------------------------------------
# Pre-built geopolitical game factories
# ---------------------------------------------------------------------------

def make_war_of_attrition(
    v1_range: tuple[float, float] = (2.0, 6.0),
    c1_range: tuple[float, float] = (1.0, 4.0),
    v2_range: tuple[float, float] = (2.0, 5.0),
    c2_range: tuple[float, float] = (2.0, 4.0),
    n_states: int = 3,
) -> GameSpec:
    """
    War of Attrition (Chapter 4, Table 4.2-4.3).
    Two central banks decide each round: MAINTAIN EASY POLICY or HALT EASING.
    Values (V) and costs (C) are drawn from ranges each round.

    State 0 = both easing, State 1 = p1 gave up, State 2 = p2 gave up.
    Actions: 0 = maintain, 1 = halt
    """
    def payoff_fn(state, a1, a2, rng):
        v1 = rng.uniform(*v1_range)
        c1 = rng.uniform(*c1_range)
        v2 = rng.uniform(*v2_range)
        c2 = rng.uniform(*c2_range)

        if state == 0:          # both still easing
            if a1 == 0 and a2 == 0:   # both maintain
                return v1 - c1, v2 - c2, 0
            elif a1 == 1 and a2 == 0:  # p1 halts first
                return 0.0, v2, 2
            elif a1 == 0 and a2 == 1:  # p2 halts first
                return v1, 0.0, 1
            else:                       # both halt simultaneously
                return 0.0, 0.0, 0
        else:
            return 0.0, 0.0, state     # terminal states

    return GameSpec(
        n_states=n_states,
        n_actions_p1=2, n_actions_p2=2,
        payoff_fn=payoff_fn,
        action_names_p1=["Maintain Easy Policy", "Halt Easing"],
        action_names_p2=["Maintain Easy Policy", "Halt Easing"],
        state_names=["Both Easing", "P1 Gave Up", "P2 Gave Up"],
    )


def make_trade_war(
    pursuer_speeds: list[float] | None = None,
    evader_speeds: list[float] | None = None,
    n_states: int = 10,
) -> GameSpec:
    """
    Pursuit-Evasion Trade War (Chapter 4, Section 4.6).
    Pursuer tries to close export-volume gap; Evader tries to maintain lead.
    Actions: 0=reduce tariffs, 1=weaken currency, 2=export subsidies, 3=tax rebates.
    """
    pursuer_speeds = pursuer_speeds or [1.6, 1.8, 1.4, 1.2]
    evader_speeds  = evader_speeds  or [1.5, 1.7, 1.6, 1.3]

    def payoff_fn(state, a1, a2, rng):
        noise = rng.normal(0, 0.1)
        ps = pursuer_speeds[a1] + noise
        es = evader_speeds[a2] + noise
        gap_change = ps - es
        next_state = int(np.clip(state + np.sign(gap_change), 0, n_states - 1))

        # Pursuer reward: bonus if caught up (state → high), penalty if behind
        r1 = 100.0 if next_state == n_states - 1 else 10.0 - (n_states - 1 - next_state)
        # Evader reward: bonus for maintaining lead
        r2 = 100.0 if next_state == 0 else next_state * 2.0
        return r1, r2, next_state

    return GameSpec(
        n_states=n_states,
        n_actions_p1=4, n_actions_p2=4,
        payoff_fn=payoff_fn,
        action_names_p1=["Reduce Tariffs", "Weaken Currency",
                         "Export Subsidies", "Tax Rebates"],
        action_names_p2=["Reduce Tariffs", "Weaken Currency",
                         "Export Subsidies", "Tax Rebates"],
    )