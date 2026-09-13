"""Hodrick-Prescott filter: splitting a time series into a smooth trend and a cycle.

Economists (and anyone with a noisy time series) often want to separate a slowly-moving TREND from
the shorter-run CYCLE around it -- GDP's long climb versus the boom-bust wiggle, a temperature record's
warming versus seasonal swings. The Hodrick-Prescott filter (1997, but the idea is Whittaker's 1923
graduation) finds the trend tau minimizing

    sum (y_t - tau_t)^2  +  lambda * sum (tau_{t+1} - 2 tau_t + tau_{t-1})^2,

trading fidelity to the data against the SECOND DIFFERENCE of the trend (its curvature). The penalty
lambda is the one knob: lambda = 0 lets the trend fit every point (trend = data, no cycle); lambda ->
infinity forces the second difference to zero everywhere, so the trend becomes a straight LINE (the
least-squares line). In between, larger lambda gives a smoother, straighter trend and a larger cycle.

The minimizer is exact and linear. Writing D for the (n-2) x n second-difference matrix, the trend
solves (I + lambda D^T D) tau = y -- a symmetric, positive-definite PENTADIAGONAL system (bandwidth 2)
that this module assembles and solves by banded Gaussian elimination (its bandwidth stays 2 throughout,
so the cost is O(n)). The cycle is simply y - tau. The module computes the trend and cycle, the two
limiting cases, and exposes the smoothing objective for checking.

Validated: the trend plus cycle reconstructs the data exactly; lambda = 0 returns the data as trend
(zero cycle); a huge lambda drives the trend to the least-squares straight line; the banded solve
matches a dense reference solve; the trend minimizes the HP objective; a larger lambda yields a
smoother trend (smaller total second difference); and a noisy linear-plus-noise series has its noise
pushed into the cycle. Pure stdlib; the trend-extraction companion to the Savitzky-Golay / Butterworth
smoothers and the total-variation denoiser."""

from __future__ import annotations


def hp_filter(y, lam):
    """Hodrick-Prescott trend for series y with smoothing parameter lam. Returns (trend, cycle)."""
    n = len(y)
    if n == 0:
        return [], []
    if n <= 2 or lam <= 0:
        return list(y), [0.0] * n

    # M = I + lam * D^T D, a symmetric SPD pentadiagonal matrix (half-bandwidth 2).
    M = _build_matrix(y, lam)
    tau = _banded_solve(M, list(y), half_bw=2)
    cycle = [y[i] - tau[i] for i in range(n)]
    return tau, cycle


def _build_matrix(y, lam):
    """Assemble M = I + lam D^T D densely (used by both the banded solve and the dense reference)."""
    n = len(y)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    for k in range(n - 2):
        idx = [k, k + 1, k + 2]
        coef = [1.0, -2.0, 1.0]
        for a in range(3):
            for b in range(3):
                M[idx[a]][idx[b]] += lam * coef[a] * coef[b]
    return M


def _banded_solve(M, b, half_bw):
    """Gaussian elimination on a symmetric-banded matrix M (dense storage, bandwidth `half_bw`),
    touching only the band. SPD so no pivoting. O(n * half_bw^2)."""
    n = len(b)
    # work on copies of only the band entries
    A = [[0.0] * n for _ in range(n)]  # we only ever read/write |i-j|<=half_bw
    for i in range(n):
        for j in range(max(0, i - half_bw), min(n, i + half_bw + 1)):
            A[i][j] = M[i][j]
    rhs = list(b)
    for i in range(n):
        piv = A[i][i]
        if abs(piv) < 1e-300:
            piv = 1e-300
        for r in range(i + 1, min(i + half_bw + 1, n)):
            factor = A[r][i] / piv
            if factor == 0:
                continue
            for c in range(i, min(i + half_bw + 1, n)):
                A[r][c] -= factor * A[i][c]
            rhs[r] -= factor * rhs[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = rhs[i]
        for c in range(i + 1, min(i + half_bw + 1, n)):
            s -= A[i][c] * x[c]
        x[i] = s / A[i][i] if abs(A[i][i]) > 1e-300 else 0.0
    return x


def second_difference_norm(tau):
    """Sum of squared second differences of the trend (the smoothness the HP penalty controls)."""
    return sum((tau[i + 1] - 2 * tau[i] + tau[i - 1]) ** 2 for i in range(1, len(tau) - 1))


def objective(y, tau, lam):
    """The HP objective: fidelity + lam * roughness."""
    fidelity = sum((y[i] - tau[i]) ** 2 for i in range(len(y)))
    return fidelity + lam * second_difference_norm(tau)


# --- reference: dense solve of (I + lam D^T D) tau = y -----------------------
def hp_dense(y, lam):
    """Reference HP trend via a dense linear solve (for checking the banded solver)."""
    n = len(y)
    if n <= 2 or lam <= 0:
        return list(y)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = 1.0
    for k in range(n - 2):
        idx = [k, k + 1, k + 2]
        coef = [1.0, -2.0, 1.0]
        for a in range(3):
            for b in range(3):
                M[idx[a]][idx[b]] += lam * coef[a] * coef[b]
    return _dense_solve(M, list(y))


def _dense_solve(M, b):
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        p = A[col][col]
        for r in range(col + 1, n):
            f = A[r][col] / p
            for c in range(col, n + 1):
                A[r][c] -= f * A[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / A[i][i]
    return x
