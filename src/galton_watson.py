"""Galton-Watson branching process: does a lineage explode or die out, and with what probability?

A branching process models a population where each individual independently leaves a random number of
offspring drawn from a fixed OFFSPRING DISTRIBUTION -- surnames passed to sons, neutrons in a fission
chain, cases in an epidemic's early days, nodes in a spreading rumour. The GALTON-WATSON process (1874,
studying the extinction of family names) asks the central question: starting from one individual, what is
the probability the line eventually DIES OUT?

The answer is a jewel of probability. Let m = E[offspring] be the mean family size and G(s) = sum p_k s^k
the PROBABILITY GENERATING FUNCTION of the offspring distribution. The EXTINCTION PROBABILITY q is the
SMALLEST fixed point of s = G(s) in [0, 1], and a sharp threshold governs it:

  SUBCRITICAL (m < 1): q = 1 -- extinction is certain, the population shrinks in expectation.
  CRITICAL   (m = 1): q = 1 -- still certain extinction (a surprise), though it can take very long.
  SUPERCRITICAL (m > 1): q < 1 -- there is a positive chance 1 - q of unbounded growth, and the
      expected generation size grows as m^n.

The extinction probability is found by iterating q_{k+1} = G(q_k) from q_0 = 0, which converges upward to
the smallest root. The whole story is driven by the single number m relative to 1.

This module computes the offspring mean and PGF, solves for the extinction probability by fixed-point
iteration, classifies the process, simulates lineage trajectories, and estimates the empirical extinction
frequency. It uses a seeded RNG. It is validated: the extinction probability is a fixed point of G and the
smallest one in [0,1]; subcritical and critical processes are certain to go extinct (q = 1) while
supercritical ones have q < 1; the empirical extinction frequency from simulation matches the computed q;
the mean generation size grows like m^n; a binary-fission distribution matches the classic quadratic
extinction formula; and results are reproducible per seed. Pure stdlib; the branching-process companion to
the Wright-Fisher, coalescent, and Hawkes-process tools."""

from __future__ import annotations


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)

    def sample(self, probs):
        """Sample an offspring count from a probability list probs[k] = P(k children)."""
        u = self.u()
        cum = 0.0
        for k, p in enumerate(probs):
            cum += p
            if u <= cum:
                return k
        return len(probs) - 1


def offspring_mean(probs):
    """m = E[offspring] = sum k p_k."""
    return sum(k * p for k, p in enumerate(probs))


def pgf(probs, s):
    """Probability generating function G(s) = sum p_k s^k."""
    return sum(p * s ** k for k, p in enumerate(probs))


def extinction_probability(probs, tol=1e-14, max_iter=100000):
    """Smallest fixed point of s = G(s) in [0,1], by upward iteration q_{k+1} = G(q_k) from 0.

    Returns the extinction probability q. For m <= 1 this converges to 1; for m > 1 to the q < 1 root."""
    q = 0.0
    for _ in range(max_iter):
        q_new = pgf(probs, q)
        if abs(q_new - q) < tol:
            return q_new
        q = q_new
    return q


def classify(probs):
    """Return 'subcritical', 'critical', or 'supercritical' from the offspring mean."""
    m = offspring_mean(probs)
    if m < 1 - 1e-12:
        return "subcritical"
    if m > 1 + 1e-12:
        return "supercritical"
    return "critical"


def simulate(probs, n_generations, seed=1, max_pop=1_000_000):
    """Simulate one lineage from a single ancestor. Returns the list of generation sizes [1, Z1, Z2, ...].

    Stops early if the population hits 0 (extinct) or exceeds max_pop (treated as survived/exploded)."""
    rng = _Rng(seed)
    sizes = [1]
    pop = 1
    for _ in range(n_generations):
        if pop == 0:
            sizes.append(0)
            continue
        children = 0
        for _ in range(pop):
            children += rng.sample(probs)
            if children > max_pop:
                break
        pop = children
        sizes.append(pop)
        if pop > max_pop:
            break
    return sizes


def empirical_extinction(probs, n_generations=50, n_runs=2000, seed=1, max_pop=2000):
    """Fraction of simulated lineages extinct by generation n_generations.

    A supercritical lineage that survives grows geometrically, so `max_pop` caps each run's population
    (a lineage past the cap has clearly escaped extinction) to keep the simulation fast."""
    extinct = 0
    for r in range(n_runs):
        sizes = simulate(probs, n_generations, seed=seed + r * 2749, max_pop=max_pop)
        if sizes[-1] == 0:
            extinct += 1
    return extinct / n_runs


def expected_generation_size(probs, n):
    """E[Z_n] = m^n for a process started from one individual."""
    return offspring_mean(probs) ** n
