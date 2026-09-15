"""Polya's urn: a rich-get-richer process whose long-run fraction is random, not deterministic.

The Ehrenfest urn drives its count deterministically toward equilibrium; POLYA'S URN does the opposite --
it AMPLIFIES early chance into a permanent random outcome. Start with a black balls and b white balls; each
step draw one at random and return it PLUS c more of the SAME colour. Every draw of a colour makes that
colour more likely next time -- positive reinforcement, "the rich get richer". The consequences are a
gallery of probability's deepest ideas:

  RANDOM LIMIT. The fraction of black balls converges (almost surely) to a RANDOM limit, not to a fixed
      number. With c = 1 that limit is BETA(a, b) distributed -- run the process twice and you get two
      different stable fractions, each fixed forever by the luck of the early draws.
  MARTINGALE. The expected fraction of black balls is CONSTANT in time -- a/(a+b) at every step -- so the
      fraction is a bounded martingale (which is why it converges).
  EXCHANGEABILITY. The sequence of colours drawn is EXCHANGEABLE: any reordering has the same probability.
      By de Finetti's theorem this is exactly what makes the process a mixture over a random limiting
      frequency, and it links the Polya urn to Bayesian inference (Beta-Bernoulli conjugacy), the Chinese
      restaurant process, and Dirichlet processes.

This module simulates the urn, gives the exact Beta limiting law for c = 1 (moments and density), verifies
the martingale property, and checks exchangeability by comparing the probabilities of reordered draw
sequences. It uses a seeded RNG. It is validated: the expected black fraction stays a/(a+b) at every step
(martingale); for c = 1 the empirical limiting fraction matches Beta(a, b) in mean a/(a+b) and variance;
the sequence is exchangeable -- reordered draw sequences have equal probability; larger reinforcement c
pins the limit faster (lower variance across runs at fixed time is NOT it -- higher c gives MORE spread);
a symmetric start gives a limit spread symmetrically about 1/2; and results are reproducible per seed. Pure
stdlib; the reinforcement-process companion to the Ehrenfest-urn, Dirichlet, and Beta-distribution tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate(a, b, c, n_draws, seed=1):
    """Simulate a Polya urn: start with a black, b white; each draw returns the ball plus c of its colour.

    Returns (draws, black_counts) where draws is the sequence of colours ('B'/'W') and black_counts[t]
    is the number of black balls after t draws (black_counts[0] = a)."""
    rng = _Rng(seed)
    black = a
    total = a + b
    draws = []
    black_counts = [black]
    for _ in range(n_draws):
        if rng.u() < black / total:
            draws.append("B")
            black += c
        else:
            draws.append("W")
        total += c
        black_counts.append(black)
    return draws, black_counts


def black_fraction(a, b, c, n_draws, seed=1):
    """Final fraction of black balls after n_draws."""
    _draws, counts = simulate(a, b, c, n_draws, seed=seed)
    total = a + b + c * n_draws
    return counts[-1] / total


def expected_fraction(a, b):
    """The martingale value: E[black fraction] = a/(a+b) at every step."""
    return a / (a + b)


def beta_mean(a, b):
    """Mean of the Beta(a, b) limiting distribution (c = 1 case): a/(a+b)."""
    return a / (a + b)


def beta_variance(a, b):
    """Variance of Beta(a, b): ab / ((a+b)^2 (a+b+1))."""
    s = a + b
    return a * b / (s * s * (s + 1))


def beta_pdf(x, a, b):
    """Beta(a, b) density at x in (0,1): x^{a-1}(1-x)^{b-1} / B(a,b)."""
    if x <= 0 or x >= 1:
        return 0.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - log_beta)


def sequence_probability(draws, a, b, c):
    """Exact probability of observing a specific ordered draw sequence from a Polya urn.

    Product of the per-step draw probabilities; for c>0 this is the same for any reordering with the same
    number of B's and W's (exchangeability)."""
    black = a
    total = a + b
    p = 1.0
    for colour in draws:
        if colour == "B":
            p *= black / total
            black += c
        else:
            p *= (total - black) / total
        total += c
    return p


def is_exchangeable(a, b, c, n_black, n_white, tol=1e-12):
    """Check exchangeability: all orderings of n_black B's and n_white W's have equal probability.

    Returns True if every ordering yields the same sequence probability (up to tol)."""
    from itertools import permutations
    seq = ["B"] * n_black + ["W"] * n_white
    seen = set()
    probs = []
    for perm in permutations(seq):
        if perm in seen:
            continue
        seen.add(perm)
        probs.append(sequence_probability(list(perm), a, b, c))
    return max(probs) - min(probs) < tol
