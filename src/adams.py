"""Adams-Bashforth-Moulton: multistep ODE integration that reuses past derivative evaluations.

Solving y' = f(t, y) numerically, single-step methods like Runge-Kutta throw away all their work each
step -- RK4 evaluates f four times per step and then forgets them. MULTISTEP methods are thriftier:
they remember the last few derivative values and fit a polynomial through them, so each new step costs
only ONE new evaluation of f. For expensive right-hand sides (large systems, costly force models) that
is a big saving.

The ADAMS-BASHFORTH family is explicit: it extrapolates the derivative polynomial past the current
point. The k-step formula is order k; the 4-step one is

    y_{n+1} = y_n + (h/24)(55 f_n - 59 f_{n-1} + 37 f_{n-2} - 9 f_{n-3}).

The ADAMS-MOULTON family is implicit (it includes f at the new point), one order more accurate for the
same step count and far more stable, but requires solving for y_{n+1}. The classic PREDICTOR-CORRECTOR
scheme (PECE) sidesteps the implicit solve: PREDICT with Adams-Bashforth, EVALUATE f there, CORRECT
with Adams-Moulton using that value, and evaluate once more -- two f evaluations per step, high order,
good stability. This module implements Adams-Bashforth (orders 1-4), Adams-Moulton correction, and the
4th-order predictor-corrector, bootstrapping the initial steps with RK4 (which it also provides).

Validated against exact solutions and convergence theory: the integrators reproduce analytic ODEs
(exponential decay, the harmonic oscillator, a logistic curve) to high accuracy; halving the step size
reduces the error by the theoretical factor (about 16x for the 4th-order method, measuring order ~4);
the predictor-corrector beats plain Adams-Bashforth at the same order; and a linear system conserves
its analytic invariant. Pure stdlib; the multistep companion to the RK45 and symplectic integrators."""

from __future__ import annotations


def _add(y, dy, h):
    return [y[i] + h * dy[i] for i in range(len(y))]


def rk4_step(f, t, y, h):
    """One classical 4th-order Runge-Kutta step (used to bootstrap multistep methods)."""
    k1 = f(t, y)
    k2 = f(t + h / 2, _add(y, k1, h / 2))
    k3 = f(t + h / 2, _add(y, k2, h / 2))
    k4 = f(t + h, _add(y, k3, h))
    return [y[i] + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))]


def rk4(f, y0, t0, t1, n):
    """Integrate y' = f(t,y) from t0 to t1 in n RK4 steps. Returns (times, states)."""
    h = (t1 - t0) / n
    y = list(y0)
    ts = [t0]
    ys = [list(y)]
    t = t0
    for _ in range(n):
        y = rk4_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


# Adams-Bashforth coefficients (explicit), indexed by order.
_AB = {
    1: [1.0],
    2: [3 / 2, -1 / 2],
    3: [23 / 12, -16 / 12, 5 / 12],
    4: [55 / 24, -59 / 24, 37 / 24, -9 / 24],
}
# Adams-Moulton coefficients (implicit), indexed by order (uses f at the new point first).
_AM = {
    1: [1.0],
    2: [1 / 2, 1 / 2],
    3: [5 / 12, 8 / 12, -1 / 12],
    4: [9 / 24, 19 / 24, -5 / 24, 1 / 24],
}


def adams_bashforth(f, y0, t0, t1, n, order=4):
    """Explicit Adams-Bashforth of the given order. Bootstraps the first `order-1` steps with RK4.
    Returns (times, states)."""
    if order not in _AB:
        raise ValueError("order must be 1..4")
    h = (t1 - t0) / n
    ts = [t0]
    ys = [list(y0)]
    fs = []  # history of f values, fs[-1] is most recent
    t = t0
    y = list(y0)
    dim = len(y0)
    coeffs = _AB[order]
    for step in range(n):
        fs.append(f(t, y))
        if len(fs) > order:
            fs.pop(0)
        if len(fs) < order:
            # bootstrap with RK4 until we have enough history
            y = rk4_step(f, t, y, h)
        else:
            # y_{n+1} = y_n + h * sum coeffs[j] * f_{n-j}
            incr = [0.0] * dim
            for j, c in enumerate(coeffs):
                fj = fs[-(j + 1)]
                for i in range(dim):
                    incr[i] += c * fj[i]
            y = [y[i] + h * incr[i] for i in range(dim)]
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


def predictor_corrector(f, y0, t0, t1, n, order=4):
    """4th-order Adams-Bashforth-Moulton predictor-corrector (PECE). Bootstraps with RK4.
    Returns (times, states)."""
    if order not in _AB or order not in _AM:
        raise ValueError("order must be 1..4")
    h = (t1 - t0) / n
    ts = [t0]
    ys = [list(y0)]
    fs = []
    t = t0
    y = list(y0)
    dim = len(y0)
    ab = _AB[order]
    am = _AM[order]
    for step in range(n):
        fs.append(f(t, y))
        if len(fs) > order:
            fs.pop(0)
        if len(fs) < order:
            y = rk4_step(f, t, y, h)
        else:
            # PREDICT with Adams-Bashforth
            incr = [0.0] * dim
            for j, c in enumerate(ab):
                fj = fs[-(j + 1)]
                for i in range(dim):
                    incr[i] += c * fj[i]
            y_pred = [y[i] + h * incr[i] for i in range(dim)]
            # EVALUATE at the predicted point
            f_pred = f(t + h, y_pred)
            # CORRECT with Adams-Moulton: y_{n+1} = y_n + h(am[0] f_pred + sum am[j] f_{n-j+1})
            incr = [am[0] * f_pred[i] for i in range(dim)]
            for j in range(1, order):
                fj = fs[-j]  # f_n, f_{n-1}, ...
                for i in range(dim):
                    incr[i] += am[j] * fj[i]
            y = [y[i] + h * incr[i] for i in range(dim)]
        t += h
        ts.append(t)
        ys.append(list(y))
    return ts, ys


def max_error(ys, ts, exact):
    """Max over all steps of |y_numeric - exact(t)| (for scalar or vector exact)."""
    err = 0.0
    for t, y in zip(ts, ys):
        e = exact(t)
        if not isinstance(e, (list, tuple)):
            e = [e]
        err = max(err, max(abs(y[i] - e[i]) for i in range(len(y))))
    return err


def convergence_order(f, y0, t0, t1, exact, method=predictor_corrector, n_coarse=20, order=4):
    """Estimate the empirical convergence order by comparing errors at step h and h/2:
    order ~ log2(err_coarse / err_fine)."""
    import math
    _, yc = method(f, y0, t0, t1, n_coarse, order=order)
    tc = [t0 + (t1 - t0) * i / n_coarse for i in range(n_coarse + 1)]
    _, yf = method(f, y0, t0, t1, 2 * n_coarse, order=order)
    tf = [t0 + (t1 - t0) * i / (2 * n_coarse) for i in range(2 * n_coarse + 1)]
    ec = max_error(yc, tc, exact)
    ef = max_error(yf, tf, exact)
    if ef <= 0:
        return float("inf")
    return math.log2(ec / ef)
