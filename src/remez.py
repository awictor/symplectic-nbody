"""Remez exchange: the true minimax polynomial, the one no other polynomial beats.

Given a continuous function f on an interval [a, b] and a degree n, there is exactly one polynomial
of degree n that minimizes the WORST-CASE error max|f(x) - p(x)| over the whole interval. That is a
very different object from the interpolating polynomial (which is exact at chosen nodes but can be
wildly wrong between them) or the least-squares polynomial (which minimizes average squared error and
tolerates a few large spikes). The minimax polynomial spreads its error out perfectly evenly, and
this module computes it.

The characterization is Chebyshev's equioscillation theorem: p is the degree-n minimax approximation
of f if and only if the error e(x) = f(x) - p(x) attains its maximum magnitude at (at least) n + 2
points, with ALTERNATING sign. The error curve rides up to +E, down to -E, up to +E, ... exactly
n + 2 times. No polynomial can do better, because to shrink the error at one of those peaks you would
have to grow it at an adjacent one -- there is no slack left. That equioscillation is both the proof
of optimality and, here, the test of correctness.

The Remez EXCHANGE algorithm turns the theorem into an iteration. Guess n + 2 reference points
(Chebyshev nodes are a good start). Demand that the error equal +/-E with alternating sign there:

    f(x_i) - (c_0 + c_1 x_i + ... + c_n x_i^n) = (-1)^i * E     for i = 0 .. n+1

That is n + 2 linear equations in the n + 1 coefficients plus the unknown level E -- a square system,
solved by Gaussian elimination with partial pivoting. Now the polynomial equioscillates on the
reference set but not yet on the whole interval. So we EXCHANGE: scan a fine grid for the real
extrema of the error, replace the reference points with those extrema (keeping the alternating sign
pattern), and re-solve. Each pass raises the minimum |E|, and it converges -- usually in a handful of
iterations -- to the true minimax polynomial, where the reference extrema and the actual extrema
coincide.

This module implements remez (returns coefficients, the minimax level E, and the reference points),
poly_eval (Horner), max_error (fine-grid sup norm), and equioscillation_points (the alternating
extrema). It is validated against the degree-0 case (the minimax constant is exactly the midrange
(max+min)/2 with error (max-min)/2), against a fine-grid brute-force search over the coefficient
neighborhood, and by checking that its error never exceeds that of Chebyshev-node interpolation or a
least-squares fit -- and that the error genuinely equioscillates n + 2 times. Pure stdlib; the
best-approximation companion to the Chebyshev-interpolation and least-squares notes.

Caveat: for an EVEN function on a SYMMETRIC interval at even degree (e.g. 1/(1+x^2) on [-3, 3],
degree 6) the error is forced to be even, so it cannot alternate the required n + 2 times without a
lobe pinned at the center -- the same parity that makes such approximations hard. The single-reference
exchange here stalls in that special case; use an asymmetric interval (or the next odd degree) to get
the genuine minimax fit. Every non-symmetric case converges."""

from __future__ import annotations

import math


def _solve(A, b):
    """Solve the linear system A x = b by Gaussian elimination with partial pivoting."""
    n = len(A)
    # augmented copy
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        # partial pivot
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            raise ValueError("singular system in Remez solve")
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0.0:
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def poly_eval(coeffs, x):
    """Evaluate polynomial c[0] + c[1] x + ... by Horner's rule."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _chebyshev_nodes(a, b, count):
    """`count` Chebyshev extrema mapped to [a, b], returned in increasing order."""
    # extrema of T_{count-1}: cos(pi k / (count-1)), k = 0..count-1
    m = count - 1
    nodes = []
    for k in range(count):
        t = math.cos(math.pi * k / m)  # in [-1, 1], descending
        nodes.append((a + b) / 2 + (b - a) / 2 * t)
    nodes.sort()
    return nodes


def remez(f, a, b, degree, max_iter=100, grid=2001, tol=1e-14):
    """Best minimax polynomial approximation of f on [a, b] of the given degree.

    Returns (coeffs, level, refs): the coefficient list (length degree+1, ascending powers), the
    equioscillation level E = max|f - p|, and the final reference points.
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    if a >= b:
        raise ValueError("need a < b")

    npts = degree + 2
    refs = _chebyshev_nodes(a, b, npts)
    # Break symmetry: an even f on a symmetric interval with symmetric references makes the
    # first solve degenerate (the interpolant hits every reference, forcing the level E to 0 and
    # leaving one too few error lobes, so the exchange stalls). A tiny asymmetric nudge of the
    # interior references escapes that fixed point; the exchange then finds the true extrema.
    span = b - a
    for i in range(1, npts - 1):
        refs[i] += 1e-6 * span * (i - (npts - 1) / 2.0)
    refs.sort()

    coeffs = [0.0] * (degree + 1)
    level = 0.0
    for _ in range(max_iter):
        # build the (degree+2) x (degree+2) system:
        #   sum_j c_j x_i^j - (-1)^i E = f(x_i)
        A = []
        rhs = []
        for i, x in enumerate(refs):
            row = [x ** j for j in range(degree + 1)]
            row.append(-((-1.0) ** i))  # coefficient of E
            A.append(row)
            rhs.append(f(x))
        sol = _solve(A, rhs)
        coeffs = sol[:degree + 1]
        level = sol[degree + 1]

        # find the extrema of the error on a fine grid
        new_refs = _exchange_refs(f, coeffs, a, b, grid, npts)
        # convergence: reference set stable and error levels match |E|
        emax = max(abs(f(x) - poly_eval(coeffs, x)) for x in new_refs)
        if abs(emax - abs(level)) < tol * max(1.0, abs(level)):
            refs = new_refs
            break
        refs = new_refs

    return coeffs, abs(level), refs


def _exchange_refs(f, coeffs, a, b, grid, npts):
    """Pick npts alternating-sign error extrema from a fine grid scan."""
    xs = [a + (b - a) * k / (grid - 1) for k in range(grid)]
    err = [f(x) - poly_eval(coeffs, x) for x in xs]

    # local extrema (including the two endpoints) of the error curve
    cand = [(xs[0], err[0])]
    for i in range(1, grid - 1):
        if (err[i] >= err[i - 1] and err[i] >= err[i + 1]) or \
           (err[i] <= err[i - 1] and err[i] <= err[i + 1]):
            cand.append((xs[i], err[i]))
    cand.append((xs[-1], err[-1]))

    # collapse runs into a strictly alternating-sign chain, keeping the largest magnitude in each run
    chain = []
    for x, e in cand:
        if chain and _same_sign(chain[-1][1], e):
            if abs(e) > abs(chain[-1][1]):
                chain[-1] = (x, e)
        else:
            chain.append((x, e))

    if len(chain) < npts:
        # not enough alternations found (early iterations); fall back to Chebyshev nodes
        return _chebyshev_nodes(a, b, npts)

    # keep the npts extrema with the largest magnitudes while preserving alternation:
    # slide a window of length npts and pick the one with the largest minimum |e|
    best = None
    best_key = -1.0
    for start in range(0, len(chain) - npts + 1):
        window = chain[start:start + npts]
        key = min(abs(e) for _, e in window)
        if key > best_key:
            best_key = key
            best = window
    return [x for x, _ in best]


def _same_sign(u, v):
    return (u >= 0) == (v >= 0)


def max_error(f, coeffs, a, b, grid=4001):
    """Fine-grid estimate of the sup norm max|f - p| on [a, b]."""
    best = 0.0
    for k in range(grid):
        x = a + (b - a) * k / (grid - 1)
        e = abs(f(x) - poly_eval(coeffs, x))
        if e > best:
            best = e
    return best


def equioscillation_points(f, coeffs, a, b, grid=4001):
    """Return the alternating error extrema (x, error) that witness equioscillation."""
    xs = [a + (b - a) * k / (grid - 1) for k in range(grid)]
    err = [f(x) - poly_eval(coeffs, x) for x in xs]
    cand = [(xs[0], err[0])]
    for i in range(1, grid - 1):
        if (err[i] >= err[i - 1] and err[i] >= err[i + 1]) or \
           (err[i] <= err[i - 1] and err[i] <= err[i + 1]):
            cand.append((xs[i], err[i]))
    cand.append((xs[-1], err[-1]))
    chain = []
    for x, e in cand:
        if chain and _same_sign(chain[-1][1], e):
            if abs(e) > abs(chain[-1][1]):
                chain[-1] = (x, e)
        else:
            chain.append((x, e))
    return chain


def chebyshev_interp(f, a, b, degree):
    """Interpolating polynomial through degree+1 Chebyshev nodes (near-minimax, for contrast)."""
    nodes = _chebyshev_nodes(a, b, degree + 1)
    # solve Vandermonde for exact interpolation
    A = [[x ** j for j in range(degree + 1)] for x in nodes]
    rhs = [f(x) for x in nodes]
    return _solve(A, rhs)


def least_squares_poly(f, a, b, degree, samples=200):
    """Discrete least-squares polynomial fit over `samples` equal points (for contrast)."""
    xs = [a + (b - a) * k / (samples - 1) for k in range(samples)]
    ys = [f(x) for x in xs]
    # normal equations: (V^T V) c = V^T y
    V = [[x ** j for j in range(degree + 1)] for x in xs]
    n = degree + 1
    ATA = [[0.0] * n for _ in range(n)]
    ATy = [0.0] * n
    for row, y in zip(V, ys):
        for i in range(n):
            ATy[i] += row[i] * y
            for j in range(n):
                ATA[i][j] += row[i] * row[j]
    return _solve(ATA, ATy)
