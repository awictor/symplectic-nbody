"""Levenberg-Marquardt: the workhorse of nonlinear least-squares curve fitting.

Fitting a nonlinear model to data -- a decaying exponential to a sensor trace, a Gaussian peak to a
spectrum, orbital elements to observations -- means choosing parameters p to minimize the sum of
squared residuals S(p) = sum_i r_i(p)^2, where r_i = model(x_i, p) - y_i. Two classic descent methods
each fail in a complementary way. GRADIENT DESCENT on S is robust far from the optimum but crawls in
narrow curved valleys. GAUSS-NEWTON, which approximates the Hessian by J^T J from the Jacobian J of
the residuals, converges quadratically near the optimum but can diverge wildly when that approximation
is poor. Levenberg-Marquardt (Levenberg 1944, Marquardt 1963) BLENDS them with a damping parameter
lambda:

    (J^T J + lambda * diag(J^T J)) delta = -J^T r

Large lambda makes the step a small, safe gradient-descent move; small lambda makes it a bold
Gauss-Newton leap. The algorithm ADAPTS lambda by trust: after each proposed step, if the sum of
squares went DOWN, accept it and shrink lambda (trust Gauss-Newton more); if it went UP, reject the
step and grow lambda (fall back toward gradient descent). Marquardt's refinement -- scaling the damping
by the diagonal of J^T J rather than the identity -- makes the method invariant to the units of each
parameter, which is why LM is what SciPy, gnuplot, and every instrument-calibration routine reach for.

This module implements LM with a numerical (finite-difference) Jacobian by default or a user-supplied
analytic one, the adaptive damping loop, and the recovered covariance estimate. The normal equations
are solved with the repo's LU solver. It is validated against ground truth: it recovers the exact
parameters of noiseless exponential, Gaussian, and sinusoidal models from perturbed starts; it reaches
the analytic least-squares solution on a LINEAR model (where LM must agree with the normal equations
exactly); the sum of squared residuals decreases monotonically over accepted steps; an analytic
Jacobian gives the same fit as the finite-difference one; and it drives the Rosenbrock residuals to
zero at (1, 1). Pure stdlib; the nonlinear-least-squares companion to the BFGS / L-BFGS optimizers and
the linear-regression tools."""

from __future__ import annotations

import math

from linsolve import solve


def _jacobian(residual, params, m, eps=1e-7):
    """Finite-difference Jacobian: J[i][j] = d r_i / d p_j, shape m x n."""
    n = len(params)
    r0 = residual(params)
    J = [[0.0] * n for _ in range(m)]
    for j in range(n):
        pj = list(params)
        h = eps * (abs(params[j]) + eps)
        pj[j] += h
        rj = residual(pj)
        for i in range(m):
            J[i][j] = (rj[i] - r0[i]) / h
    return J


def _JT_J_and_JT_r(J, r):
    """Return (J^T J, J^T r) for the normal equations."""
    m = len(J)
    n = len(J[0])
    JTJ = [[0.0] * n for _ in range(n)]
    JTr = [0.0] * n
    for a in range(n):
        for b in range(n):
            s = 0.0
            for i in range(m):
                s += J[i][a] * J[i][b]
            JTJ[a][b] = s
        sr = 0.0
        for i in range(m):
            sr += J[i][a] * r[i]
        JTr[a] = sr
    return JTJ, JTr


def sum_of_squares(r):
    return sum(v * v for v in r)


def levenberg_marquardt(residual, p0, m=None, jacobian=None, max_iter=200,
                        tol=1e-12, lambda0=1e-3, nu=10.0):
    """Minimize sum_i residual(p)_i^2 over parameters p, starting from p0.

    residual(p) returns the length-m residual vector. m is inferred from residual(p0) if omitted.
    jacobian(p) may supply an analytic Jacobian (m x n); otherwise finite differences are used.
    Returns a dict with 'params', 'cost' (sum of squares), 'iterations', 'converged', 'history'.
    """
    params = [float(v) for v in p0]
    r = residual(params)
    if m is None:
        m = len(r)
    n = len(params)
    lam = lambda0
    S = sum_of_squares(r)
    history = [S]
    converged = False

    for it in range(1, max_iter + 1):
        J = jacobian(params) if jacobian else _jacobian(residual, params, m)
        JTJ, JTr = _JT_J_and_JT_r(J, r)

        # try steps, growing lambda until one reduces the cost (or we give up this iteration)
        improved = False
        for _ in range(30):
            # augment the diagonal: (J^T J + lam * diag(J^T J)) delta = -J^T r
            A = [row[:] for row in JTJ]
            for d in range(n):
                A[d][d] += lam * (JTJ[d][d] if JTJ[d][d] != 0 else 1.0)
            rhs = [-JTr[d] for d in range(n)]
            try:
                delta = solve(A, rhs)
            except (ValueError, ZeroDivisionError):
                lam *= nu
                continue
            trial = [params[i] + delta[i] for i in range(n)]
            r_trial = residual(trial)
            S_trial = sum_of_squares(r_trial)
            if S_trial < S:
                # accept, decrease damping (trust Gauss-Newton more)
                params = trial
                r = r_trial
                improvement = S - S_trial
                S = S_trial
                lam = max(lam / nu, 1e-15)
                improved = True
                history.append(S)
                if improvement < tol * max(1.0, S) or S < tol:
                    converged = True
                break
            else:
                # reject, increase damping (fall back toward gradient descent)
                lam *= nu
                if lam > 1e14:
                    break
        if not improved or converged:
            break

    return {
        "params": params,
        "cost": S,
        "iterations": it,
        "converged": converged,
        "history": history,
        "covariance": _covariance(residual, params, m, jacobian, S),
    }


def _covariance(residual, params, m, jacobian, S):
    """Parameter covariance estimate sigma^2 (J^T J)^{-1}, sigma^2 = S / (m - n)."""
    n = len(params)
    if m <= n:
        return None
    J = jacobian(params) if jacobian else _jacobian(residual, params, m)
    JTJ, _ = _JT_J_and_JT_r(J, r=[0.0] * m)
    # invert JTJ column by column via the LU solver
    try:
        cov = []
        sigma2 = S / (m - n)
        for j in range(n):
            e = [1.0 if k == j else 0.0 for k in range(n)]
            col = solve([row[:] for row in JTJ], e)
            cov.append(col)
        # cov currently columns; scale and transpose to standard layout
        C = [[sigma2 * cov[j][i] for j in range(n)] for i in range(n)]
        return C
    except (ValueError, ZeroDivisionError):
        return None


def make_residual(model, xs, ys):
    """Build a residual function r_i(p) = model(x_i, p) - y_i for data (xs, ys)."""
    def residual(p):
        return [model(xs[i], p) - ys[i] for i in range(len(xs))]
    return residual


# ---- common models -----------------------------------------------------------------------------

def exp_model(x, p):
    """p[0] * exp(p[1] * x) + p[2]."""
    return p[0] * math.exp(p[1] * x) + p[2]


def gaussian_model(x, p):
    """p[0] * exp(-((x - p[1])^2) / (2 p[2]^2))."""
    return p[0] * math.exp(-((x - p[1]) ** 2) / (2 * p[2] ** 2))


def sine_model(x, p):
    """p[0] * sin(p[1] * x + p[2])."""
    return p[0] * math.sin(p[1] * x + p[2])
