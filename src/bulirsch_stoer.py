"""Bulirsch-Stoer: extreme-accuracy ODE integration by modified midpoint + Richardson extrapolation.

For a smooth ordinary differential equation y' = f(t, y), the Bulirsch-Stoer method delivers accuracy
that Runge-Kutta cannot touch. The idea combines two classical tools. First, the MODIFIED MIDPOINT
rule crosses a large step H using n internal substeps -- and crucially, its error is a power series in
the SUBSTEP SIZE h = H/n containing only EVEN powers of h. Second, take that step with a growing
sequence of substep counts n = 2, 4, 6, 8, ..., getting a family of estimates of y(t+H) with error
O(h^2), O(h^4), ... and RICHARDSON-EXTRAPOLATE them to h = 0. Because only even powers appear, each
extrapolation column jumps two orders, so a handful of midpoint sweeps reach accuracy equivalent to a
very high-order method -- often 1e-13 in a single step of a smooth problem.

The extrapolation is a polynomial-in-h^2 fit evaluated at h = 0, built as a Neville-style tableau on
the successive estimates. This module implements a single Bulirsch-Stoer step (with the sequence of
substep counts), a fixed-step driver, and helpers, keeping everything vector-valued so it solves
systems (a planet's position and velocity, a chain of coupled oscillators).

It is validated against exact solutions: it integrates y' = y (exponential), the harmonic oscillator
(sine/cosine), and a two-body Kepler orbit to far higher accuracy than a comparable-cost RK step; the
extrapolation genuinely raises the order (error shrinks super-fast with the substep sequence length);
the harmonic oscillator's energy is conserved to near machine precision over many periods; and it
agrees with the repo's RK45 solver where both are accurate. Pure stdlib; the high-accuracy ODE
companion to the RK45, Adams, and symplectic-integrator tools, and a direct application of Richardson
extrapolation to differential equations."""

from __future__ import annotations


def _axpy(a, x, y):
    """Return a*x + y for vectors x, y."""
    return [a * x[i] + y[i] for i in range(len(x))]


def modified_midpoint(f, t, y, H, n):
    """Advance y over a step H using n substeps of the modified midpoint rule. Returns y(t+H) estimate."""
    h = H / n
    y0 = list(y)
    # z_0 = y, z_1 = z_0 + h f(t, z_0)
    z_prev = list(y0)
    fy = f(t, y0)
    z_curr = _axpy(h, fy, y0)
    for m in range(1, n):
        tm = t + m * h
        fz = f(tm, z_curr)
        z_next = _axpy(2 * h, fz, z_prev)  # z_{m+1} = z_{m-1} + 2h f(t_m, z_m)
        z_prev, z_curr = z_curr, z_next
    # final smoothing: y(t+H) ~ (z_n + z_{n-1} + h f(t+H, z_n)) / 2
    f_end = f(t + H, z_curr)
    return [0.5 * (z_curr[i] + z_prev[i] + h * f_end[i]) for i in range(len(y))]


def bulirsch_stoer_step(f, t, y, H, k_max=8):
    """One Bulirsch-Stoer step over H: modified midpoint at n=2,4,6,... + Richardson extrapolation.

    Returns (y_new, error_estimate). Extrapolates the even-power error series in the substep size.
    """
    # substep sequence (even numbers): n_k = 2, 4, 6, 8, ...
    ns = [2 * (k + 1) for k in range(k_max)]
    dim = len(y)
    # Neville-style extrapolation tableau on estimates T[k], with x_k = (H/n_k)^2
    T = []
    xs = []
    for k in range(k_max):
        est = modified_midpoint(f, t, y, H, ns[k])
        xs.append((H / ns[k]) ** 2)
        row = [est]
        for j in range(1, k + 1):
            # Neville rational/poly extrapolation to x=0:
            # T[k][j] = T[k][j-1] + (T[k][j-1] - T[k-1][j-1]) / ((xs[k-j]/xs[k]) - 1)
            ratio = xs[k - j] / xs[k]
            denom = ratio - 1.0
            prev_same = row[j - 1]
            prev_up = T[k - 1][j - 1]
            new = [prev_same[i] + (prev_same[i] - prev_up[i]) / denom for i in range(dim)]
            row.append(new)
        T.append(row)
    best = T[-1][-1]
    # error estimate: difference between the two best extrapolants
    if k_max >= 2:
        second = T[-1][-2]
        err = max(abs(best[i] - second[i]) for i in range(dim))
    else:
        err = float("inf")
    return best, err


def solve_fixed(f, t0, y0, t1, steps=100, k_max=8):
    """Integrate y' = f(t, y) from t0 to t1 in `steps` equal Bulirsch-Stoer steps.

    Returns (ts, ys) with the solution sampled at each step boundary.
    """
    H = (t1 - t0) / steps
    t = t0
    y = list(y0)
    ts = [t]
    ys = [list(y)]
    for _ in range(steps):
        y, _err = bulirsch_stoer_step(f, t, y, H, k_max=k_max)
        t += H
        ts.append(t)
        ys.append(list(y))
    return ts, ys


def solve(f, t0, y0, t1, k_max=8):
    """Convenience: integrate to t1 and return the final state (100 fixed steps)."""
    _ts, ys = solve_fixed(f, t0, y0, t1, steps=100, k_max=k_max)
    return ys[-1]


# ---- reference systems ------------------------------------------------------------------------

def harmonic_oscillator(omega=1.0):
    """f for y'' = -omega^2 y as a first-order system [x', v'] = [v, -omega^2 x]."""
    def f(t, y):
        return [y[1], -omega * omega * y[0]]
    return f


def kepler_2d(mu=1.0):
    """f for a 2-D Kepler orbit: state [x, y, vx, vy], acceleration -mu r / |r|^3."""
    def f(t, s):
        x, y, vx, vy = s
        r3 = (x * x + y * y) ** 1.5
        return [vx, vy, -mu * x / r3, -mu * y / r3]
    return f


def harmonic_energy(state, omega=1.0):
    """Energy of the harmonic oscillator state [x, v]: 0.5 v^2 + 0.5 omega^2 x^2."""
    x, v = state
    return 0.5 * v * v + 0.5 * omega * omega * x * x


def kepler_energy(state, mu=1.0):
    """Specific orbital energy of a Kepler state [x, y, vx, vy]: 0.5 v^2 - mu / r."""
    x, y, vx, vy = state
    r = (x * x + y * y) ** 0.5
    return 0.5 * (vx * vx + vy * vy) - mu / r
