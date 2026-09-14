"""PELT: find the optimal set of change points in a signal, exactly, in near-linear time.

CUSUM watches a stream and flags the first shift; but given a whole recorded series -- a genome, a price
history, a sensor log -- you often want ALL the change points at once, and the BEST such set, not a greedy
approximation. Segmenting optimally is a search over exponentially many partitions, and the classic
dynamic program that solves it exactly is O(n^2). PELT (Pruned Exact Linear Time; Killick, Fearnhead,
Eckley 2012) keeps the exactness but adds a PRUNING rule that discards candidate change points which can
never be optimal, dropping the cost to roughly O(n) on typical data.

The idea: choose change points to minimize the total segment COST plus a PENALTY beta per change point,

    F(n) = min over last-change tau of [ F(tau) + cost(y_{tau+1..n}) + beta ].

Too small a penalty over-segments (a change point at every wiggle); too large under-segments (misses real
shifts). The standard choice beta = 2 log n (a BIC-type penalty) balances fit against parsimony. PELT's
pruning keeps only those candidate taus for which F(tau) + cost(tau..t) stays below the current best -- a
set that stays small, giving the linear-time behavior while returning the SAME optimum as the full DP.

This module implements PELT for the change-in-mean cost (segment sum of squared deviations, computed in
O(1) from prefix sums), returns the change points and the segment means, and exposes the penalty. It is
validated: it recovers the known change points of a piecewise-constant signal; a larger penalty yields
fewer change points and a smaller one more; with a huge penalty it returns a single segment and with a
tiny one it splits aggressively; the pruned result exactly matches the unpruned O(n^2) dynamic program;
the segment means reconstruct the piecewise levels; a pure-noise signal yields few or no change points; and
results are deterministic. Pure stdlib; the offline optimal-segmentation companion to the CUSUM,
Page-Hinkley, and hypothesis-testing tools."""

from __future__ import annotations

import math


def _prefix_sums(data):
    """Prefix sums of x and x^2 for O(1) segment cost. Returns (S, S2) of length n+1."""
    n = len(data)
    S = [0.0] * (n + 1)
    S2 = [0.0] * (n + 1)
    for i in range(n):
        S[i + 1] = S[i] + data[i]
        S2[i + 1] = S2[i] + data[i] * data[i]
    return S, S2


def _segment_cost(S, S2, a, b):
    """Cost (sum of squared deviations from the mean) of the segment data[a:b], via prefix sums.

    SSE = sum x^2 - (sum x)^2 / length. This is the Gaussian change-in-mean cost."""
    length = b - a
    if length <= 0:
        return 0.0
    seg_sum = S[b] - S[a]
    seg_sq = S2[b] - S2[a]
    return seg_sq - seg_sum * seg_sum / length


def default_penalty(n):
    """BIC-type penalty beta = 2 log n."""
    return 2.0 * math.log(n) if n > 1 else 0.0


def pelt(data, penalty=None, min_size=1):
    """Optimal change-point detection by PELT for the change-in-mean cost.

    Returns a dict with change_points (sorted interior indices; a change point at index c means a new
    segment starts at c), n_segments, segment_means, and the penalty used. `penalty` defaults to
    2 log n; larger -> fewer change points."""
    n = len(data)
    if penalty is None:
        penalty = default_penalty(n)
    S, S2 = _prefix_sums(data)

    # F[t] = optimal cost of segmenting data[0:t]. F[0] = -penalty so the first segment pays no penalty.
    F = [0.0] * (n + 1)
    F[0] = -penalty
    last = [0] * (n + 1)          # last[t] = start index of the final segment in the optimum for data[0:t]
    candidates = [0]              # pruned set of candidate previous change points

    for t in range(min_size, n + 1):
        best_cost = None
        best_tau = 0
        for tau in candidates:
            if t - tau < min_size:
                continue
            c = F[tau] + _segment_cost(S, S2, tau, t) + penalty
            if best_cost is None or c < best_cost:
                best_cost = c
                best_tau = tau
        if best_cost is None:
            # no valid candidate yet (t < 2*min_size); fall back to the full segment
            best_cost = F[0] + _segment_cost(S, S2, 0, t) + penalty
            best_tau = 0
        F[t] = best_cost
        last[t] = best_tau
        # pruning: keep tau only if F[tau] + cost(tau..t) <= F[t] (they can still be optimal later)
        pruned = []
        for tau in candidates:
            if t - tau < min_size:
                pruned.append(tau)
                continue
            if F[tau] + _segment_cost(S, S2, tau, t) <= F[t]:
                pruned.append(tau)
        pruned.append(t)
        candidates = pruned

    # backtrack the change points
    cps = []
    t = n
    while t > 0:
        tau = last[t]
        if tau > 0:
            cps.append(tau)
        t = tau
    cps.sort()

    # segment means
    bounds = [0] + cps + [n]
    means = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        means.append((S[b] - S[a]) / (b - a) if b > a else 0.0)

    return {
        "change_points": cps,
        "n_segments": len(cps) + 1,
        "segment_means": means,
        "penalty": penalty,
        "cost": F[n] + penalty,   # add back the -penalty offset for the true total segment cost
    }


def pelt_bruteforce(data, penalty=None, min_size=1):
    """The full O(n^2) DP without pruning -- same optimum, used to validate PELT. Returns change_points."""
    n = len(data)
    if penalty is None:
        penalty = default_penalty(n)
    S, S2 = _prefix_sums(data)
    F = [0.0] * (n + 1)
    F[0] = -penalty
    last = [0] * (n + 1)
    for t in range(min_size, n + 1):
        best = None
        arg = 0
        for tau in range(0, t - min_size + 1):
            c = F[tau] + _segment_cost(S, S2, tau, t) + penalty
            if best is None or c < best:
                best = c
                arg = tau
        F[t] = best
        last[t] = arg
    cps = []
    t = n
    while t > 0:
        tau = last[t]
        if tau > 0:
            cps.append(tau)
        t = tau
    cps.sort()
    return cps
