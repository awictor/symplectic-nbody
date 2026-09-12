"""Markov decision processes: value iteration and policy iteration for optimal control.

A MARKOV DECISION PROCESS is the mathematical frame for sequential decision-making under uncertainty
-- a robot navigating, an inventory being restocked, a game being played. It has STATES, ACTIONS,
stochastic TRANSITIONS (taking action a in state s lands in state s' with some probability), REWARDS,
and a DISCOUNT FACTOR gamma that trades immediate against future reward. The goal is a POLICY -- a
choice of action in every state -- that maximizes the expected discounted return. MDPs are the
foundation of reinforcement learning and optimal control.

Two classic dynamic-programming algorithms solve an MDP exactly. VALUE ITERATION repeatedly applies
the BELLMAN OPTIMALITY equation: the value of a state is the best over actions of the immediate
reward plus gamma times the expected value of the next state, V(s) <- max_a sum_s' P(s'|s,a)[R + gamma
V(s')]. This is a CONTRACTION mapping (it shrinks the gap to the true values by a factor gamma each
sweep), so it converges geometrically to the unique optimal value function, from which the greedy
policy is optimal. POLICY ITERATION alternates two steps: POLICY EVALUATION solves the linear system
for the current policy's exact values, then POLICY IMPROVEMENT makes the policy greedy with respect to
them; because there are finitely many policies and each round strictly improves, it converges to the
optimum in a small number of iterations, often faster than value iteration.

This module implements both algorithms for a finite MDP given as transition and reward tables, plus a
gridworld builder for the canonical navigation example. It is verified against exact references: that
value and policy iteration converge to the SAME optimal value function and policy, that the result
satisfies the Bellman optimality equation (the value function is a fixed point), that a known
gridworld yields the expected shortest-path-to-goal policy, that a higher discount makes the agent
value distant rewards more, and that value iteration's error contracts by the discount factor each
sweep. Pure stdlib; a reinforcement-learning companion to the HMM, Kalman-filter, and
dynamic-programming notes."""

from __future__ import annotations


class MDP:
    """A finite Markov decision process.

    states: list of state ids. actions: list of action ids. transitions[s][a] = list of
    (probability, next_state, reward) tuples. gamma: discount factor in [0, 1)."""

    def __init__(self, states, actions, transitions, gamma=0.9):
        self.states = list(states)
        self.actions = list(actions)
        self.P = transitions
        self.gamma = gamma

    def _q_value(self, s, a, V):
        """The action-value Q(s,a) = sum_s' P(s'|s,a) [reward + gamma V(s')]."""
        total = 0.0
        for prob, s2, reward in self.P[s].get(a, []):
            total += prob * (reward + self.gamma * V[s2])
        return total

    def value_iteration(self, tol=1e-10, max_iter=10000):
        """Bellman value iteration. Returns (V, policy, iterations)."""
        V = {s: 0.0 for s in self.states}
        for it in range(1, max_iter + 1):
            delta = 0.0
            newV = {}
            for s in self.states:
                actions = list(self.P[s].keys())
                if not actions:
                    newV[s] = 0.0
                    continue
                best = max(self._q_value(s, a, V) for a in actions)
                newV[s] = best
                delta = max(delta, abs(best - V[s]))
            V = newV
            if delta < tol:
                break
        policy = self._greedy_policy(V)
        return V, policy, it

    def _greedy_policy(self, V):
        policy = {}
        for s in self.states:
            actions = list(self.P[s].keys())
            if not actions:
                policy[s] = None
                continue
            policy[s] = max(actions, key=lambda a: self._q_value(s, a, V))
        return policy

    def evaluate_policy(self, policy, tol=1e-10, max_iter=10000):
        """Iterative policy evaluation: the exact value of following `policy`."""
        V = {s: 0.0 for s in self.states}
        for _ in range(max_iter):
            delta = 0.0
            for s in self.states:
                a = policy[s]
                if a is None:
                    continue
                v = self._q_value(s, a, V)
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < tol:
                break
        return V

    def policy_iteration(self, max_iter=1000):
        """Policy iteration: alternate evaluation and greedy improvement. Returns (V, policy, iters)."""
        # start with an arbitrary policy (first available action)
        policy = {s: (list(self.P[s].keys())[0] if self.P[s] else None) for s in self.states}
        for it in range(1, max_iter + 1):
            V = self.evaluate_policy(policy)
            stable = True
            for s in self.states:
                actions = list(self.P[s].keys())
                if not actions:
                    continue
                best = max(actions, key=lambda a: self._q_value(s, a, V))
                if best != policy[s]:
                    policy[s] = best
                    stable = False
            if stable:
                return V, policy, it
        return self.evaluate_policy(policy), policy, max_iter

    def bellman_residual(self, V):
        """Max over states of |V(s) - max_a Q(s,a)| -- zero when V is the optimal fixed point."""
        res = 0.0
        for s in self.states:
            actions = list(self.P[s].keys())
            if not actions:
                continue
            best = max(self._q_value(s, a, V) for a in actions)
            res = max(res, abs(V[s] - best))
        return res


def gridworld(width, height, goal, obstacles=None, step_reward=-0.04, goal_reward=1.0,
              gamma=0.9, slip=0.0):
    """Build a gridworld MDP. States are (x, y) cells; actions are 'N','S','E','W'. The goal is
    absorbing. `slip` is the probability the agent moves perpendicular instead of intended (stochastic
    dynamics); 0 gives deterministic moves."""
    obstacles = set(obstacles or [])
    states = [(x, y) for x in range(width) for y in range(height) if (x, y) not in obstacles]
    actions = ["N", "S", "E", "W"]
    moves = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
    perp = {"N": ["E", "W"], "S": ["E", "W"], "E": ["N", "S"], "W": ["N", "S"]}

    def step(s, mv):
        nx, ny = s[0] + moves[mv][0], s[1] + moves[mv][1]
        if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
            return (nx, ny)
        return s                              # bump into a wall/obstacle -> stay

    transitions = {}
    for s in states:
        transitions[s] = {}
        if s == goal:
            continue                          # absorbing: no actions leave the goal
        for a in actions:
            outcomes = {}
            # intended move with prob (1 - slip), perpendicular slips split the rest
            main_p = 1.0 - slip
            dest = step(s, a)
            outcomes[dest] = outcomes.get(dest, 0.0) + main_p
            for pa in perp[a]:
                d = step(s, pa)
                outcomes[d] = outcomes.get(d, 0.0) + slip / 2
            transitions[s][a] = [
                (p, s2, goal_reward if s2 == goal else step_reward)
                for s2, p in outcomes.items()
            ]
    return MDP(states, actions, transitions, gamma=gamma)
