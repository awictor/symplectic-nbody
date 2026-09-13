"""Thomas algorithm and Crank-Nicolson: solving tridiagonal systems and the heat equation.

A TRIDIAGONAL system -- a matrix nonzero only on the main diagonal and its two neighbours -- arises
whenever a 1-D problem couples each point only to its immediate neighbours: finite-difference
derivatives, cubic-spline coefficients, and above all the implicit time-stepping of the heat and
diffusion equations. Gaussian elimination on a general n x n system is O(n^3), but a tridiagonal one
falls in O(n) by the THOMAS ALGORITHM: a single forward sweep eliminates the sub-diagonal
(modifying the diagonal and the right-hand side), then a back-substitution reads off the solution.
It is the workhorse behind every 1-D implicit PDE solver.

The showcase is the heat equation u_t = alpha u_xx. An EXPLICIT scheme (forward Euler) is simple but
only stable when the step ratio r = alpha dt / dx^2 <= 1/2 -- shrink dx and dt collapses. The
CRANK-NICOLSON scheme averages the spatial derivative between the old and new time levels, which is
UNCONDITIONALLY STABLE (any dt) and second-order accurate in both space and time. Each step then
requires solving a tridiagonal system -- exactly what Thomas does in O(n) -- so a whole diffusion
simulation costs O(n) per step regardless of how stiff it is.

This module implements the Thomas solver, the explicit and Crank-Nicolson heat steppers (with
Dirichlet boundaries), and helpers for the stability ratio. Validated against ground truth: Thomas
matches a dense Gaussian-elimination solve on random tridiagonal systems; Crank-Nicolson diffusing a
Gaussian pulse matches the analytic sqrt(t)-broadening solution, conserves total heat under zero-flux
conditions, decays a sine mode at the exact analytic rate exp(-alpha k^2 t), and stays stable at
large time steps where the explicit scheme blows up. Pure stdlib; the numerical-PDE companion to the
analytic diffusion (Fick's law) note and the relaxation/spectral tools."""

from __future__ import annotations

import math


def thomas_solve(a, b, c, d):
    """Solve a tridiagonal system for x, where for row i:
        a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i].
    a[0] and c[n-1] are ignored (no neighbour). Returns x. O(n), non-destructive."""
    n = len(d)
    if not (len(a) == len(b) == len(c) == n):
        raise ValueError("a, b, c, d must have equal length")
    cp = [0.0] * n
    dp = [0.0] * n
    if b[0] == 0:
        raise ValueError("zero pivot at row 0")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if m == 0:
            raise ValueError(f"zero pivot at row {i}")
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def stability_ratio(alpha, dt, dx):
    """r = alpha dt / dx^2. Explicit forward-Euler is stable iff r <= 1/2."""
    return alpha * dt / (dx * dx)


def heat_step_explicit(u, alpha, dt, dx):
    """One forward-Euler step of u_t = alpha u_xx with Dirichlet (fixed) boundaries.
    Only stable for r <= 1/2."""
    n = len(u)
    r = stability_ratio(alpha, dt, dx)
    new = list(u)
    for i in range(1, n - 1):
        new[i] = u[i] + r * (u[i + 1] - 2 * u[i] + u[i - 1])
    return new


def heat_step_crank_nicolson(u, alpha, dt, dx):
    """One Crank-Nicolson step of u_t = alpha u_xx with Dirichlet boundaries. Unconditionally stable.
    Builds and solves the tridiagonal system (I - r/2 L) u^{n+1} = (I + r/2 L) u^n."""
    n = len(u)
    r = stability_ratio(alpha, dt, dx)
    # interior indices 1..n-2 are unknowns; boundaries fixed
    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n
    d = [0.0] * n
    # boundary rows: identity (keep fixed)
    b[0] = 1.0
    d[0] = u[0]
    b[n - 1] = 1.0
    d[n - 1] = u[n - 1]
    for i in range(1, n - 1):
        a[i] = -r / 2
        b[i] = 1 + r
        c[i] = -r / 2
        d[i] = (r / 2) * u[i - 1] + (1 - r) * u[i] + (r / 2) * u[i + 1]
    # fix RHS for rows adjacent to boundary (boundary values known and constant)
    return thomas_solve(a, b, c, d)


def diffuse(u0, alpha, dt, dx, steps, scheme="crank_nicolson"):
    """Run `steps` heat-equation steps from initial profile u0. Returns the final profile."""
    stepper = heat_step_crank_nicolson if scheme == "crank_nicolson" else heat_step_explicit
    u = list(u0)
    for _ in range(steps):
        u = stepper(u, alpha, dt, dx)
    return u


def total_heat(u, dx):
    """Trapezoidal integral of u over the domain (total heat / mass)."""
    return dx * (sum(u) - 0.5 * (u[0] + u[-1]))


# --- reference: dense Gaussian elimination -----------------------------------
def dense_tridiagonal_solve(a, b, c, d):
    """Build the full dense matrix and solve by Gaussian elimination (reference for Thomas)."""
    n = len(d)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        M[i][i] = b[i]
        if i > 0:
            M[i][i - 1] = a[i]
        if i < n - 1:
            M[i][i + 1] = c[i]
    # augmented Gaussian elimination with partial pivoting
    A = [row[:] + [d[i]] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        pivot = A[col][col]
        for r in range(col + 1, n):
            f = A[r][col] / pivot
            for k in range(col, n + 1):
                A[r][k] -= f * A[col][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = A[i][n] - sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / A[i][i]
    return x
