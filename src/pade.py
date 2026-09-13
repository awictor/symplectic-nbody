"""Pade approximants: rational functions that beat Taylor series, especially near poles.

A Taylor series approximates a function by a polynomial, but polynomials cannot capture a POLE (a
blow-up like 1/(1-x)) and often converge only in a small disk. A PADE APPROXIMANT [m/n] is a RATIONAL
function P(x)/Q(x), with deg P = m and deg Q = n, whose own Taylor expansion agrees with the target's
through order m+n. Because it can put zeros in the denominator, it reproduces poles, extends the range
of accurate approximation far beyond the Taylor radius, and often converges where the series diverges
-- the workhorse behind function evaluation in libraries, analytic continuation, and resummation of
divergent perturbation series in physics.

Given the Taylor coefficients c_0, c_1, ..., c_{m+n} of f, the [m/n] approximant is found by a linear
system. Writing Q(x) = 1 + q_1 x + ... + q_n x^n and P(x) = p_0 + ... + p_m x^m, matching
f(x) Q(x) = P(x) through order m+n gives, for the coefficients of orders m+1..m+n (where P has no
terms), a linear system for the q's:

    sum_{j=1}^{n} c_{m+k-j} q_j = -c_{m+k},   k = 1..n,

and then the p's fall out directly: p_k = c_k + sum_{j=1}^{min(k,n)} c_{k-j} q_j. This module builds
the approximant from Taylor coefficients (or from a function via finite-difference derivatives),
evaluates P/Q, and reports the numerator/denominator coefficients.

Validated: the Pade approximant's own series matches the input Taylor coefficients through order m+n;
[m/0] reduces to the Taylor polynomial; the [1/1] approximant of 1/(1-x) is exact (it recovers the
pole); Pade approximants of exp, log(1+x), and arctan are far more accurate than the same-order Taylor
polynomial over a wide interval; and a function with a pole is tracked past the Taylor radius. Pure
stdlib; the rational-approximation companion to the barycentric / Chebyshev interpolation and the
sequence-acceleration tools."""

from __future__ import annotations


def _solve(A, b):
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(b)
    if n == 0:
        return []
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        p = M[col][col]
        if abs(p) < 1e-15:
            continue
        for r in range(col + 1, n):
            f = M[r][col] / p
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def pade(coeffs, m, n):
    """Build the [m/n] Pade approximant from Taylor coefficients coeffs[0..m+n].
    Returns (P, Q): numerator coefficients (length m+1) and denominator (length n+1, Q[0]=1)."""
    if len(coeffs) < m + n + 1:
        raise ValueError(f"need at least {m + n + 1} Taylor coefficients")
    c = list(coeffs)

    # denominator: solve sum_{j=1}^n c_{m+k-j} q_j = -c_{m+k}, k=1..n
    if n > 0:
        A = [[0.0] * n for _ in range(n)]
        b = [0.0] * n
        for k in range(1, n + 1):
            b[k - 1] = -c[m + k]
            for j in range(1, n + 1):
                idx = m + k - j
                A[k - 1][j - 1] = c[idx] if idx >= 0 else 0.0
        q = _solve(A, b)
    else:
        q = []
    Q = [1.0] + q  # Q[0] = 1

    # numerator: p_k = c_k + sum_{j=1}^{min(k,n)} c_{k-j} q_j
    P = [0.0] * (m + 1)
    for k in range(m + 1):
        s = c[k]
        for j in range(1, min(k, n) + 1):
            s += c[k - j] * q[j - 1]
        P[k] = s
    return P, Q


def evaluate(P, Q, x):
    """Evaluate the rational function P(x)/Q(x) by Horner's method."""
    num = 0.0
    for coef in reversed(P):
        num = num * x + coef
    den = 0.0
    for coef in reversed(Q):
        den = den * x + coef
    return num / den


def pade_series(P, Q, order):
    """The Taylor coefficients of P(x)/Q(x) up to `order` (for checking the matching property)."""
    # long division: c_k such that P = Q * (sum c_k x^k)
    c = [0.0] * (order + 1)
    for k in range(order + 1):
        pk = P[k] if k < len(P) else 0.0
        s = pk
        for j in range(1, min(k, len(Q) - 1) + 1):
            s -= Q[j] * c[k - j]
        c[k] = s / Q[0]
    return c


def taylor_coeffs_from_function(f, order, x0=0.0, h=1e-3):
    """Estimate Taylor coefficients c_k = f^(k)(x0)/k! by finite differences (small order only)."""
    import math
    # central finite-difference stencils for derivatives 0..order
    coeffs = []
    for k in range(order + 1):
        # k-th derivative via the standard central difference of order k
        deriv = 0.0
        # use the finite-difference formula with binomial weights
        for i in range(k + 1):
            deriv += (-1) ** i * math.comb(k, i) * f(x0 + (k / 2 - i) * h)
        deriv /= h ** k
        coeffs.append(deriv / math.factorial(k))
    return coeffs
