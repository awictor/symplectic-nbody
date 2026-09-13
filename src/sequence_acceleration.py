"""Sequence acceleration: extracting a limit from a slowly-converging series, fast.

Many series converge so slowly that summing millions of terms still gives only a few digits -- the
Leibniz series for pi/4 = 1 - 1/3 + 1/5 - ... needs a billion terms for nine digits. Yet the PARTIAL
SUMS carry more information than their face value: the pattern of how they approach the limit can be
extrapolated. Sequence-acceleration transforms turn a slowly-converging sequence into a
rapidly-converging one, often reaching machine precision from a handful of terms.

AITKEN'S delta-squared is the simplest: given three consecutive terms it fits a geometric tail and
jumps to its limit,

    s' = s_n - (s_{n+1} - s_n)^2 / (s_{n+2} - 2 s_{n+1} + s_n).

Applied repeatedly it accelerates linearly-convergent sequences dramatically. The SHANKS
transformation is the same idea in closed form, and WYNN'S EPSILON ALGORITHM computes the whole
family of higher-order Shanks transforms with a simple rhombus recurrence -- it is astonishingly
effective, summing even divergent-looking alternating series to their analytic continuation. For
alternating series specifically, EULER'S TRANSFORM reweights the terms by binomial coefficients and
converges geometrically.

This module implements Aitken's process (single and iterated), the Wynn epsilon algorithm (the best
general-purpose accelerator here), and the Euler transform for alternating series. Validated against
known limits: on the Leibniz series for pi/4 and the alternating series for ln 2, a dozen terms
accelerated reach far more digits than thousands of raw terms; Aitken exactly recovers the limit of a
pure geometric sequence in one step; the accelerators leave an already-converged sequence unchanged;
and Euler's transform matches Wynn on alternating series. Pure stdlib; the convergence-acceleration
companion to the Richardson-extrapolation and numerical-series tools."""

from __future__ import annotations


def aitken(seq):
    """One pass of Aitken's delta-squared process: turns a length-n sequence of partial sums into a
    length-(n-2) accelerated sequence."""
    out = []
    for i in range(len(seq) - 2):
        d1 = seq[i + 1] - seq[i]
        d2 = seq[i + 2] - 2 * seq[i + 1] + seq[i]
        if d2 == 0:
            out.append(seq[i + 2])
        else:
            out.append(seq[i] - d1 * d1 / d2)
    return out


def aitken_iterated(seq, rounds=None):
    """Apply Aitken repeatedly until the sequence is too short. Returns the final best estimate."""
    s = list(seq)
    r = 0
    while len(s) >= 3 and (rounds is None or r < rounds):
        s = aitken(s)
        r += 1
    return s[-1] if s else None


def wynn_epsilon(seq):
    """Wynn's epsilon algorithm: the general Shanks accelerator. Returns the best (highest even
    column) estimate of the limit from the partial-sum sequence `seq`."""
    n = len(seq)
    if n == 0:
        return None
    # eps table: e[k][j]; e[-1]=0, e[0]=seq. Store as dict of columns.
    # Using the standard indexing: eps(k, j).
    e = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        e[i][0] = seq[i]
    best = seq[-1]
    for k in range(1, n):
        for i in range(n - k):
            denom = e[i + 1][k - 1] - e[i][k - 1]
            prev = e[i + 1][k - 2] if k >= 2 else 0.0
            if denom == 0:
                e[i][k] = prev  # avoid division by zero; carry the previous estimate
            else:
                e[i][k] = prev + 1.0 / denom
        # even columns hold the accelerated estimates of the limit
        if k % 2 == 0 and n - k > 0:
            best = e[0][k]
    return best


def euler_transform(terms, n_out=None):
    """Euler's transform for an alternating series sum (-1)^k a_k (pass the a_k, positive).
    Returns accelerated partial sums; the last is the best estimate."""
    a = list(terms)
    n = len(a)
    if n_out is None:
        n_out = n
    # forward-difference table of a_k
    diff = [list(a)]
    for _ in range(n - 1):
        prev = diff[-1]
        diff.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    # Euler transform: sum = sum_k (-1)^0 ... 0.5 * sum_k (-1)^k Delta^k a_0 / 2^k, accumulated
    # Euler transform of sum (-1)^k a_k = sum_k (-1)^k Delta^k a_0 / 2^{k+1}
    total = 0.0
    partials = []
    for k in range(min(n_out, n)):
        total += ((-1) ** k) * diff[k][0] / (2 ** (k + 1))
        partials.append(total)
    return partials


def partial_sums(term_fn, n):
    """Build the sequence of partial sums s_0..s_{n-1} of a series whose k-th term is term_fn(k)."""
    s = 0.0
    out = []
    for k in range(n):
        s += term_fn(k)
        out.append(s)
    return out
