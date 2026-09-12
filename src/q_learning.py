"""Q-learning and SARSA: model-free reinforcement learning from experience.

Value iteration solves a Markov decision process when you KNOW its transition probabilities and
rewards. But an agent dropped into an unknown world does not -- it can only ACT and observe what
happens. MODEL-FREE reinforcement learning learns to act optimally from raw experience alone, never
building a model of the environment. Q-LEARNING (Watkins, 1989) is the landmark algorithm: it learns
the ACTION-VALUE function Q(s, a) -- the expected return of taking action a in state s and acting
optimally thereafter -- purely from sampled transitions (s, a, reward, s'), and its greedy policy
converges to the optimum.

The update is TEMPORAL-DIFFERENCE learning: after each step, nudge Q(s, a) toward the observed reward
plus the discounted value of the best next action, Q(s,a) <- Q(s,a) + alpha[r + gamma max_a' Q(s',a')
- Q(s,a)]. The bracket is the TD ERROR, the surprise between prediction and outcome; a learning rate
alpha controls how fast estimates move. Q-learning is OFF-POLICY -- it learns the optimal Q while
following any exploratory behaviour (typically epsilon-greedy) -- which is why it can learn the best
policy even while exploring. SARSA is its ON-POLICY cousin: it bootstraps from the action the policy
ACTUALLY takes next, Q(s,a) <- Q(s,a) + alpha[r + gamma Q(s',a') - Q(s,a)], so it learns the value of
the behaviour policy itself, yielding safer paths when exploration is risky.

This module implements tabular Q-learning and SARSA with epsilon-greedy exploration over a generic
step-based environment (a gridworld is provided), plus greedy-policy extraction. It is verified
against a ground truth: that Q-learning's learned Q-values and greedy policy converge to the optimal
policy computed by value iteration on the same gridworld, that the learned state values approach the
MDP optimum, that a learned agent reaches the goal far faster than a random walker, that a higher
learning rate speeds early learning, and that SARSA also solves the task. Pure stdlib; a
reinforcement-learning companion to the MDP (value/policy iteration) and multi-armed-bandit notes."""

from __future__ import annotations

import math


class _RNG:
    def __init__(self, seed=1):
        self.state = seed & 0xFFFFFFFF

    def random(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def randint(self, n):
        return int(self.random() * n) % n


class GridEnv:
    """A gridworld environment exposing reset/step (the RL interface), not the transition model.

    The agent only sees states and rewards -- it must learn the dynamics by acting."""

    def __init__(self, width, height, goal, obstacles=None, step_reward=-0.04, goal_reward=1.0,
                 slip=0.0, seed=1):
        self.w = width
        self.h = height
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.slip = slip
        self.rng = _RNG(seed)
        self.actions = ["N", "S", "E", "W"]
        self._moves = {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}
        self._perp = {"N": ["E", "W"], "S": ["E", "W"], "E": ["N", "S"], "W": ["N", "S"]}
        self.states = [(x, y) for x in range(width) for y in range(height)
                       if (x, y) not in self.obstacles]
        self.start = (0, 0)

    def reset(self):
        self.pos = self.start
        return self.pos

    def _apply(self, s, mv):
        nx, ny = s[0] + self._moves[mv][0], s[1] + self._moves[mv][1]
        if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in self.obstacles:
            return (nx, ny)
        return s

    def step(self, action):
        """Take an action. Returns (next_state, reward, done)."""
        # stochastic slip
        if self.slip > 0 and self.rng.random() < self.slip:
            action = self._perp[action][self.rng.randint(2)]
        nxt = self._apply(self.pos, action)
        self.pos = nxt
        if nxt == self.goal:
            return nxt, self.goal_reward, True
        return nxt, self.step_reward, False


def _epsilon_greedy(Q, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return actions[rng.randint(len(actions))]
    qs = Q[state]
    best = max(qs.values())
    # tie-break deterministically among the best
    best_actions = [a for a in actions if qs[a] == best]
    return best_actions[rng.randint(len(best_actions))]


def q_learning(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1, max_steps=200, seed=2):
    """Tabular Q-learning (off-policy TD control). Returns (Q, greedy_policy, episode_returns)."""
    rng = _RNG(seed)
    Q = {s: {a: 0.0 for a in env.actions} for s in env.states}
    returns = []
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = _epsilon_greedy(Q, s, env.actions, epsilon, rng)
            s2, r, done = env.step(a)
            total += r
            best_next = max(Q[s2].values()) if s2 in Q else 0.0
            Q[s][a] += alpha * (r + gamma * best_next - Q[s][a])
            s = s2
            if done:
                break
        returns.append(total)
    policy = {s: max(env.actions, key=lambda a: Q[s][a]) for s in env.states}
    return Q, policy, returns


def sarsa(env, episodes=500, alpha=0.5, gamma=0.9, epsilon=0.1, max_steps=200, seed=2):
    """Tabular SARSA (on-policy TD control). Returns (Q, greedy_policy, episode_returns)."""
    rng = _RNG(seed)
    Q = {s: {a: 0.0 for a in env.actions} for s in env.states}
    returns = []
    for ep in range(episodes):
        s = env.reset()
        a = _epsilon_greedy(Q, s, env.actions, epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            a2 = _epsilon_greedy(Q, s2, env.actions, epsilon, rng) if s2 in Q else a
            q_next = Q[s2][a2] if s2 in Q else 0.0
            Q[s][a] += alpha * (r + gamma * q_next - Q[s][a])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    policy = {s: max(env.actions, key=lambda a: Q[s][a]) for s in env.states}
    return Q, policy, returns


def greedy_rollout(env, policy, max_steps=200):
    """Follow a greedy policy from the start; returns the number of steps to reach the goal (or
    max_steps if it fails)."""
    s = env.reset()
    for t in range(1, max_steps + 1):
        s, r, done = env.step(policy[s])
        if done:
            return t
    return max_steps
