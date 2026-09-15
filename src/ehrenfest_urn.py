"""The Ehrenfest urn: how reversible microscopic dynamics produce irreversible macroscopic diffusion.

In 1907 Paul and Tatiana Ehrenfest proposed a toy model to resolve a paradox that had dogged statistical
mechanics: how can time-REVERSIBLE molecular collisions produce the time-IRREVERSIBLE spread of a gas and
the relentless rise of entropy (Boltzmann's H-theorem), when Poincaré's recurrence theorem says any finite
system must eventually return arbitrarily close to its start? The model: N labelled balls split between two
urns; at each step pick ONE ball uniformly at random and move it to the OTHER urn. That is all.

The chain on the count k of balls in urn A is a birth-death Markov chain: from k it goes to k-1 with
probability k/N (a ball in A is picked) or to k+1 with probability (N-k)/N. Its beautiful lessons:

  IRREVERSIBILITY FROM REVERSIBILITY. Started with all N balls in one urn, the count relaxes rapidly and
      monotonically (in expectation) toward the even split N/2 -- diffusion and rising entropy emerge --
      even though every single move is reversible.
  EQUILIBRIUM. The stationary distribution is BINOMIAL(N, 1/2): at equilibrium each ball is independently
      in either urn with probability 1/2, sharply peaked at N/2. The chain satisfies detailed balance, so
      it is reversible and this is its unique stationary law.
  RECURRENCE, BUT SLOW. Poincaré is satisfied: the system does return to the all-in-one state -- but the
      mean recurrence time is 2^N steps, astronomically long for macroscopic N, which is why we never see
      a gas spontaneously un-mix.

This module simulates the urn, gives the binomial stationary distribution, verifies detailed balance, and
computes the expected drift and mean recurrence time. It uses a seeded RNG. It is validated: the long-run
occupation histogram matches Binomial(N, 1/2); starting from all-in-one, the mean count relaxes toward N/2
and the entropy rises; the transition probabilities satisfy detailed balance against the binomial; the
expected next count moves toward N/2 (a linear restoring drift); the mean recurrence time to the all-in-one
state equals 2^N; and results are reproducible per seed. Pure stdlib; the reversible-Markov-chain companion
to the Gillespie-SSA, Markov-chain, and maxwell-boltzmann tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def simulate(n, k0, n_steps, seed=1):
    """Simulate the Ehrenfest urn: n balls, k0 initially in urn A. Returns the list of counts in A
    over n_steps+1 time points (including the start)."""
    rng = _Rng(seed)
    k = k0
    counts = [k]
    for _ in range(n_steps):
        # pick a ball uniformly; it is in A with probability k/n
        if rng.u() < k / n:
            k -= 1   # a ball leaves A
        else:
            k += 1   # a ball enters A
        counts.append(k)
    return counts


def stationary_distribution(n):
    """Binomial(n, 1/2) stationary distribution: pi_k = C(n,k) / 2^n."""
    total = 2 ** n
    return [math.comb(n, k) / total for k in range(n + 1)]


def transition_prob(n, k, k_next):
    """Transition probability from k to k_next: k/n (down) or (n-k)/n (up), else 0."""
    if k_next == k - 1:
        return k / n
    if k_next == k + 1:
        return (n - k) / n
    return 0.0


def detailed_balance_residual(n):
    """Max |pi_k P(k->k+1) - pi_{k+1} P(k+1->k)| over k -- should be ~0 (chain is reversible)."""
    pi = stationary_distribution(n)
    worst = 0.0
    for k in range(n):
        lhs = pi[k] * transition_prob(n, k, k + 1)
        rhs = pi[k + 1] * transition_prob(n, k + 1, k)
        worst = max(worst, abs(lhs - rhs))
    return worst


def expected_next(n, k):
    """E[k_{t+1} | k_t = k] = k + (n - 2k)/n -- a linear restoring drift toward n/2.

    From k, go down w.p. k/n and up w.p. (n-k)/n, so the expected change is (n-k)/n - k/n = (n-2k)/n."""
    return k + (n - 2 * k) / n


def mean_recurrence_time(n):
    """Mean recurrence time to the all-in-one state (k=0 or k=n) = 2^n (= 1/pi_0)."""
    return 2 ** n


def entropy(counts_histogram):
    """Shannon entropy (bits) of an occupation histogram (a list of counts or probabilities)."""
    total = sum(counts_histogram)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts_histogram:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def occupation_histogram(counts, n):
    """Histogram of how often each count 0..n was visited over a trajectory."""
    hist = [0] * (n + 1)
    for k in counts:
        hist[k] += 1
    return hist
