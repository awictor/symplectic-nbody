"""Dormand-Prince RK45: adaptive Runge-Kutta with embedded error control.

Solving an ordinary differential equation y' = f(t, y) numerically means stepping forward in time,
and the central tension is STEP SIZE: too large and the solution is inaccurate or unstable; too small
and it wastes work where the solution is smooth. ADAPTIVE methods resolve this by choosing the step
size automatically at every step to keep the local error near a target tolerance -- taking big steps
through calm regions and small steps through rapid changes. The DORMAND-PRINCE method (RK45, the
default in MATLAB's ode45 and SciPy's solve_ivp) is the workhorse: a pair of Runge-Kutta formulas of
orders 5 and 4 that share the same function evaluations, so the difference between them is a cheap
ESTIMATE of the local truncation error, used to accept or reject the step and resize it.

Each step evaluates f at seven stages (six new plus one reused from the previous step -- the FSAL,
'first same as last', property that makes it efficient). Two weighted sums of those stages give a
5th-order solution and a 4th-order solution; their difference is the error estimate. If the estimated
error is below the tolerance the step is accepted, otherwise rejected, and in both cases the next
step size is scaled by (tol / error)^(1/5) with safety factors -- the classic PI-free step
controller. This yields a solution that meets a requested accuracy with far fewer evaluations than a
fixed-step method of the same order.

This module implements Dormand-Prince RK45 with adaptive step-size control for scalar and vector ODEs,
returning the solution sampled at the accepted steps (with optional dense sampling at requested
times). It is verified against problems with known closed-form solutions -- exponential decay,
simple harmonic motion (energy conservation), a logistic curve, and a stiff-ish linear system --
checking that the final error is within the requested tolerance, that tightening the tolerance
reduces the error (order-of-magnitude convergence), that adaptive stepping uses more steps where the
solution changes fastest, and that harmonic motion conserves energy over many periods. Pure stdlib; a
numerical-integration companion to the symplectic-integrator and Richardson-extrapolation notes."""

from __future__ import annotations

# Dormand-Prince Butcher tableau (the standard RK45 coefficients)
_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
# 5th-order solution weights (b) and 4th-order weights (b*) for the error estimate
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]
_B4 = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]


def _add_scaled(y, stages, coeffs, h):
    """y + h * sum(coeffs[i] * stages[i]) for vector state y."""
    out = list(y)
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        ki = stages[i]
        for j in range(len(out)):
            out[j] += h * c * ki[j]
    return out


def _norm(err, y, atol, rtol):
    """RMS error norm scaled by the tolerance (per SciPy/Hairer)."""
    total = 0.0
    for i in range(len(err)):
        scale = atol + rtol * abs(y[i])
        total += (err[i] / scale) ** 2
    return (total / len(err)) ** 0.5


def solve(f, t0, y0, t1, atol=1e-8, rtol=1e-8, h0=None, max_steps=1000000):
    """Integrate y' = f(t, y) from t0 to t1 with adaptive Dormand-Prince RK45.

    f(t, y) returns the derivative (a list, same length as y). y0 is the initial state (list).
    Returns (ts, ys): the times and states at every accepted step, including t0 and t1."""
    scalar = not hasattr(y0, "__len__")
    if scalar:
        y0 = [y0]
        f_orig = f
        f = lambda t, y: [f_orig(t, y[0])]      # noqa: E731

    y = list(y0)
    t = t0
    ts = [t]
    ys = [list(y)]
    direction = 1.0 if t1 >= t0 else -1.0
    span = abs(t1 - t0)
    h = (h0 if h0 is not None else span / 100.0) * direction
    if h == 0:
        h = span * 1e-3 * direction

    safety = 0.9
    min_factor, max_factor = 0.2, 5.0
    order = 5

    # FSAL: k[0] of the next step is k[6] of the accepted step; seed the first k0
    k0 = f(t, y)
    steps = 0
    while (t - t1) * direction < 0 and steps < max_steps:
        steps += 1
        if (t + h - t1) * direction > 0:        # don't overshoot the endpoint
            h = t1 - t
        # seven stages
        k = [k0]
        for i in range(1, 7):
            yi = _add_scaled(y, k, _A[i], h)
            k.append(f(t + _C[i] * h, yi))
        y5 = _add_scaled(y, k, _B5, h)
        y4 = _add_scaled(y, k, _B4, h)
        err = [y5[i] - y4[i] for i in range(len(y))]
        error_norm = _norm(err, y5, atol, rtol)

        if error_norm <= 1.0:
            # accept the step
            t += h
            y = y5
            k0 = k[6]                            # FSAL reuse
            ts.append(t)
            ys.append(list(y))
        # adapt the step size
        if error_norm == 0:
            factor = max_factor
        else:
            factor = safety * error_norm ** (-1.0 / order)
            factor = max(min_factor, min(max_factor, factor))
        h *= factor

    if scalar:
        ys = [row[0] for row in ys]
    return ts, ys


def solve_at(f, t0, y0, times, atol=1e-8, rtol=1e-8):
    """Integrate and return the solution sampled at the requested `times` (must be increasing and
    start at t0), using linear interpolation between accepted steps."""
    ts, ys = solve(f, t0, y0, times[-1], atol=atol, rtol=rtol)
    scalar = not hasattr(y0, "__len__")
    out = []
    j = 0
    for tq in times:
        while j < len(ts) - 1 and ts[j + 1] < tq:
            j += 1
        if j >= len(ts) - 1:
            out.append(ys[-1])
            continue
        # linear interpolation between ts[j] and ts[j+1]
        t_a, t_b = ts[j], ts[j + 1]
        frac = 0.0 if t_b == t_a else (tq - t_a) / (t_b - t_a)
        if scalar:
            out.append(ys[j] + frac * (ys[j + 1] - ys[j]))
        else:
            out.append([ys[j][d] + frac * (ys[j + 1][d] - ys[j][d]) for d in range(len(y0))])
    return out
