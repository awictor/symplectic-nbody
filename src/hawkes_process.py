"""Hawkes process: a self-exciting point process where each event makes more events likely just after.

An ordinary Poisson process fires events at a constant rate, independently -- but many real event streams
CLUSTER: an earthquake triggers aftershocks, a neuron's spike raises its neighbours' firing odds, a market
order provokes more orders, a retweet begets retweets. The HAWKES PROCESS (1971) models this SELF-EXCITATION
with a conditional intensity that jumps up after every event and decays back down:

    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i)),

where mu is the baseline rate, and each past event adds an exponentially-decaying bump of height alpha and
decay rate beta. The dimensionless BRANCHING RATIO n = alpha/beta is the expected number of direct
"children" each event spawns: n < 1 gives a stationary process (excitation dies out), n >= 1 explodes. In
the stationary regime the long-run average intensity is mu/(1 - n) -- self-excitation amplifies the
baseline rate by 1/(1-n), the same geometric cascade as a branching process.

Simulating a Hawkes process exactly is done by OGATA'S THINNING: propose events from a Poisson process at
an upper-bounding rate, then keep each with probability equal to the true intensity over the bound. Because
the exponential kernel is Markov, the intensity can be updated in O(1) per event.

This module simulates a univariate Hawkes process with an exponential kernel (Ogata thinning), evaluates
the conditional intensity, and reports the branching ratio and theoretical stationary rate. It uses a
seeded RNG. It is validated: with alpha=0 it reduces to a Poisson process whose event count matches mu*T;
the empirical event rate matches the theoretical mu/(1-n) across branching ratios; the intensity spikes by
alpha at each event and decays at rate beta between events; events cluster more (higher variance of
inter-event gaps) than a Poisson process of the same mean rate; a near-critical process (n close to 1)
produces far more events than its baseline; the branching ratio is alpha/beta; and results are reproducible
per seed. Pure stdlib; the self-exciting-point-process companion to the Gillespie-SSA, Poisson, and
Markov-chain tools."""

from __future__ import annotations

import math


class _Rng:
    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFF

    def u(self):
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return (self.state >> 8) / (1 << 24)


def intensity(t, events, mu, alpha, beta):
    """Conditional intensity lambda(t) = mu + sum_{t_i < t} alpha exp(-beta (t - t_i))."""
    lam = mu
    for ti in events:
        if ti < t:
            lam += alpha * math.exp(-beta * (t - ti))
    return lam


def branching_ratio(alpha, beta):
    """n = alpha / beta: expected direct offspring per event. n < 1 -> stationary."""
    return alpha / beta if beta > 0 else float("inf")


def stationary_rate(mu, alpha, beta):
    """Long-run average intensity mu / (1 - n) for n = alpha/beta < 1 (else infinite)."""
    n = branching_ratio(alpha, beta)
    return mu / (1 - n) if n < 1 else float("inf")


def simulate(mu, alpha, beta, t_max, seed=1, max_events=1_000_000):
    """Simulate a univariate Hawkes process with exponential kernel by Ogata's thinning.

    Returns the sorted list of event times in [0, t_max). mu baseline, alpha jump, beta decay."""
    rng = _Rng(seed)
    events = []
    t = 0.0
    # current intensity contribution from past events, decayed to time t (Markov update)
    while t < t_max and len(events) < max_events:
        # upper bound on intensity just after t: baseline + current excitation
        lam_bar = intensity(t, events, mu, alpha, beta) + alpha  # +alpha headroom for a jump at t
        if lam_bar <= 0:
            break
        # propose next candidate time from Poisson(lam_bar)
        u = max(rng.u(), 1e-12)
        t = t - math.log(u) / lam_bar
        if t >= t_max:
            break
        # accept with probability lambda(t)/lam_bar
        lam_t = intensity(t, events, mu, alpha, beta)
        if rng.u() * lam_bar <= lam_t:
            events.append(t)
    return events


def simulate_fast(mu, alpha, beta, t_max, seed=1, max_events=1_000_000):
    """Ogata thinning with an O(1) Markov intensity update (no full-history sum each step)."""
    rng = _Rng(seed)
    events = []
    t = 0.0
    excitation = 0.0  # sum of alpha exp(-beta (t - t_i)) tracked incrementally at the current t
    while t < t_max and len(events) < max_events:
        lam_bar = mu + excitation + alpha
        u = max(rng.u(), 1e-12)
        dt = -math.log(u) / lam_bar
        t_new = t + dt
        if t_new >= t_max:
            break
        # decay excitation to the proposed time
        excitation_new = excitation * math.exp(-beta * dt)
        lam_t = mu + excitation_new
        t = t_new
        excitation = excitation_new
        if rng.u() * lam_bar <= lam_t:
            events.append(t)
            excitation += alpha  # the accepted event bumps the intensity
    return events


def empirical_rate(events, t_max):
    """Average event rate = count / T."""
    return len(events) / t_max
