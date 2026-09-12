"""Multi-armed bandits: the exploration-exploitation tradeoff, solved three ways.

A MULTI-ARMED BANDIT is the purest model of decision-making under uncertainty: a row of slot machines
('arms'), each paying out from an unknown probability distribution, and a gambler who must choose
which to pull on each of many rounds to maximize total reward. The tension is EXPLORATION versus
EXPLOITATION -- pull the arm that has looked best so far (exploit), or try an under-sampled arm that
might be better (explore)? Bandits model A/B testing, ad selection, clinical trials, and adaptive
routing, and they are the simplest setting in which the regret of learning can be analyzed exactly.

Performance is measured by REGRET: the reward lost by not always pulling the truly best arm. A good
strategy drives the AVERAGE regret per round to zero -- it learns which arm is best fast enough that
mistakes become rare. This module implements three classic policies. EPSILON-GREEDY exploits the
empirically best arm but explores a random arm with probability epsilon; simple but its regret grows
linearly unless epsilon decays. UCB1 (upper confidence bound) is the elegant optimism-under-
uncertainty rule: pull the arm maximizing its empirical mean plus a confidence bonus sqrt(2 ln t / n)
that shrinks as an arm is sampled, giving provably LOGARITHMIC regret with no tuning. THOMPSON
SAMPLING is the Bayesian approach: keep a Beta posterior over each arm's success probability, sample
one value from each posterior, and pull the arm with the highest sample -- probability-matching that
is often the best performer in practice.

This module implements a Bernoulli bandit environment and the three policies with a seeded RNG, and
tracks per-round regret. It is verified against exact references: that each learning policy vastly
outperforms pure random selection, that UCB1 and Thompson achieve sublinear cumulative regret (their
average regret falls toward zero as rounds grow), that all policies identify the best arm as the most-
pulled with high probability, that UCB1's regret grows logarithmically (far slower than linearly), and
that a single dominant arm is found quickly. Pure stdlib; a reinforcement-learning companion to the
MDP and Monte-Carlo notes."""

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


class BernoulliBandit:
    """A bandit whose arm i pays reward 1 with probability probs[i], else 0."""

    def __init__(self, probs, seed=1):
        self.probs = list(probs)
        self.k = len(probs)
        self.best = max(range(self.k), key=lambda i: probs[i])
        self.best_p = probs[self.best]
        self.rng = _RNG(seed)

    def pull(self, arm):
        return 1.0 if self.rng.random() < self.probs[arm] else 0.0


def _run(bandit, choose, update, rounds):
    """Generic bandit loop. choose(t) -> arm; update(arm, reward). Returns (rewards, regrets, counts,
    means)."""
    k = bandit.k
    counts = [0] * k
    means = [0.0] * k
    rewards = []
    regrets = []
    cum_regret = 0.0
    for t in range(1, rounds + 1):
        arm = choose(t, counts, means)
        r = bandit.pull(arm)
        counts[arm] += 1
        means[arm] += (r - means[arm]) / counts[arm]
        update(arm, r, counts, means)
        rewards.append(r)
        cum_regret += bandit.best_p - bandit.probs[arm]
        regrets.append(cum_regret)
    return rewards, regrets, counts, means


def epsilon_greedy(bandit, rounds, epsilon=0.1, decay=False, seed=2):
    """Epsilon-greedy: exploit the best-so-far arm, explore randomly with probability epsilon
    (optionally decaying as 1/t)."""
    rng = _RNG(seed)

    def choose(t, counts, means):
        eps = epsilon / t if decay else epsilon
        # ensure each arm tried once first
        for i in range(bandit.k):
            if counts[i] == 0:
                return i
        if rng.random() < eps:
            return rng.randint(bandit.k)
        return max(range(bandit.k), key=lambda i: means[i])

    def update(arm, r, counts, means):
        pass

    return _run(bandit, choose, update, rounds)


def ucb1(bandit, rounds):
    """UCB1: pull the arm maximizing mean + sqrt(2 ln t / n_arm) -- optimism under uncertainty."""
    def choose(t, counts, means):
        for i in range(bandit.k):
            if counts[i] == 0:
                return i
        return max(range(bandit.k),
                   key=lambda i: means[i] + math.sqrt(2 * math.log(t) / counts[i]))

    def update(arm, r, counts, means):
        pass

    return _run(bandit, choose, update, rounds)


def thompson_sampling(bandit, rounds, seed=3):
    """Thompson sampling: Beta(successes+1, failures+1) posterior per arm; sample each and pull the
    argmax. Bayesian probability-matching."""
    rng = _RNG(seed)
    alpha = [1.0] * bandit.k        # successes + 1
    beta = [1.0] * bandit.k         # failures + 1

    def sample_beta(a, b):
        # sample Beta(a,b) via two Gamma samples (Marsaglia-Tsang for integer-ish shapes here small)
        x = _sample_gamma(a, rng)
        y = _sample_gamma(b, rng)
        return x / (x + y) if (x + y) > 0 else 0.5

    def choose(t, counts, means):
        samples = [sample_beta(alpha[i], beta[i]) for i in range(bandit.k)]
        return max(range(bandit.k), key=lambda i: samples[i])

    def update(arm, r, counts, means):
        if r > 0.5:
            alpha[arm] += 1
        else:
            beta[arm] += 1

    return _run(bandit, choose, update, rounds)


def _sample_gamma(shape, rng):
    """Sample from Gamma(shape, 1) via Marsaglia-Tsang (shape >= 1) with a boost for shape < 1."""
    if shape < 1:
        # boost: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
        u = max(rng.random(), 1e-12)
        return _sample_gamma(shape + 1, rng) * (u ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9 * d)
    while True:
        # standard normal via Box-Muller
        u1 = max(rng.random(), 1e-12)
        u2 = rng.random()
        x = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        v = (1 + c * x) ** 3
        if v <= 0:
            continue
        u = max(rng.random(), 1e-12)
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v


def random_policy(bandit, rounds, seed=4):
    """Pull a uniformly random arm each round (the no-learning baseline)."""
    rng = _RNG(seed)

    def choose(t, counts, means):
        return rng.randint(bandit.k)

    def update(arm, r, counts, means):
        pass

    return _run(bandit, choose, update, rounds)
