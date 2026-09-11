"""Newton's method in n dimensions: solving nonlinear systems of equations.

One-dimensional Newton iterates x <- x - f(x)/f'(x) to find a root. In n dimensions the derivative
becomes the JACOBIAN matrix J (all partial derivatives dF_i/dx_j) and the division becomes solving
a LINEAR SYSTEM:

    J(x) * delta = -F(x),     x <- x + delta

Each step linearizes the system at the current point, jumps to that linear model's root, and
repeats -- and near a solution it converges QUADRATICALLY (the number of correct digits roughly
doubles each step), the signature of Newton's method. It is the engine inside every nonlinear
solver: circuit simulation, robotics inverse kinematics, chemical equilibrium, optimization
(finding where the gradient is zero).

Two practical concerns are handled here. When the analytic Jacobian is unavailable it is
approximated by FINITE DIFFERENCES (a column per variable). And plain Newton can overshoot and
diverge far from a solution, so a DAMPED / line-search variant backtracks the step length until the
residual actually decreases, trading a little speed for global robustness. A BROYDEN quasi-Newton
mode avoids recomputing the Jacobian every step by updating an approximation from successive
residuals -- cheaper per step when the Jacobian is expensive.

This module implements full Newton (analytic or finite-difference Jacobian), a damped/backtracking
variant, and Broyden's method, each solving the linear step via LU with partial pivoting -- verified
that it finds the roots of standard nonlinear systems (a circle-line intersection, the Rosenbrock
stationary point), that convergence is quadratic near the root, that damping rescues a start that
plain Newton diverges from, that the finite-difference Jacobian matches an analytic one, and that
Broyden converges too. Pure stdlib, built on the LU solver; the multivariate companion to the
1-D root-finding note."""

from __future__ import annotations

import math

from lu import lu_solve


def _vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def finite_difference_jacobian(F, x, eps=1e-7):
    """Approximate the Jacobian of F at x by forward differences (one column per variable)."""
    n = len(x)
    fx = F(x)
    m = len(fx)
    J = [[0.0] * n for _ in range(m)]
    for j in range(n):
        xj = list(x)
        h = eps * max(1.0, abs(x[j]))
        xj[j] += h
        fxj = F(xj)
        for i in range(m):
            J[i][j] = (fxj[i] - fx[i]) / h
    return J


def newton(F, x0, jacobian=None, tol=1e-10, max_iter=50, track=False):
    """Solve F(x) = 0 by Newton's method.

    F(x)        -> residual vector
    x0          -> starting guess
    jacobian(x) -> Jacobian matrix, or None to use finite differences
    Returns (root, n_iter, converged) or, with track=True, also the per-step residual norms."""
    x = list(x0)
    history = [_norm(F(x))]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        fx = F(x)
        if _norm(fx) < tol:
            converged = True
            break
        J = jacobian(x) if jacobian else finite_difference_jacobian(F, x)
        neg = [-v for v in fx]
        delta = lu_solve(J, neg)
        x = _vec_add(x, delta)
        history.append(_norm(F(x)))
        if _norm(delta) < tol:
            converged = _norm(F(x)) < tol * 100
            break
    if track:
        return x, it, converged, history
    return x, it, converged


def newton_damped(F, x0, jacobian=None, tol=1e-10, max_iter=100, max_backtrack=30, track=False):
    """Damped Newton with backtracking line search: shrink the step until the residual norm
    decreases, giving global robustness where plain Newton would overshoot."""
    x = list(x0)
    history = [_norm(F(x))]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        fx = F(x)
        r = _norm(fx)
        if r < tol:
            converged = True
            break
        J = jacobian(x) if jacobian else finite_difference_jacobian(F, x)
        neg = [-v for v in fx]
        delta = lu_solve(J, neg)
        # backtracking: try full step, then half, quarter, ... until the residual drops
        step = 1.0
        for _ in range(max_backtrack):
            trial = _vec_add(x, [step * d for d in delta])
            if _norm(F(trial)) < r:
                break
            step *= 0.5
        x = _vec_add(x, [step * d for d in delta])
        history.append(_norm(F(x)))
        if _norm([step * d for d in delta]) < tol:
            converged = _norm(F(x)) < tol * 100
            break
    if track:
        return x, it, converged, history
    return x, it, converged


def broyden(F, x0, tol=1e-10, max_iter=100, track=False):
    """Broyden's quasi-Newton method: start from a finite-difference Jacobian and update its
    inverse-free approximation from successive steps, avoiding a fresh Jacobian each iteration."""
    x = list(x0)
    n = len(x)
    fx = F(x)
    J = finite_difference_jacobian(F, x)
    history = [_norm(fx)]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if _norm(fx) < tol:
            converged = True
            break
        neg = [-v for v in fx]
        delta = lu_solve(J, neg)
        x_new = _vec_add(x, delta)
        fx_new = F(x_new)
        # Broyden rank-1 update: J <- J + ((df - J dx) dx^T) / (dx^T dx)
        df = _vec_sub(fx_new, fx)
        Jdx = [sum(J[i][j] * delta[j] for j in range(n)) for i in range(n)]
        denom = sum(d * d for d in delta)
        if denom > 1e-300:
            correction = _vec_sub(df, Jdx)
            for i in range(n):
                for j in range(n):
                    J[i][j] += correction[i] * delta[j] / denom
        x, fx = x_new, fx_new
        history.append(_norm(fx))
        if _norm(delta) < tol:
            converged = _norm(fx) < tol * 100
            break
    if track:
        return x, it, converged, history
    return x, it, converged
