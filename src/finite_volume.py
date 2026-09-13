"""Finite-volume shock capturing: Godunov and MUSCL schemes for scalar conservation laws.

Conservation laws u_t + f(u)_x = 0 -- advection, traffic flow, gas dynamics -- develop SHOCKS:
discontinuities that form in finite time even from smooth initial data (a fast-moving bit of the
profile catches up to a slow bit ahead of it). Naive finite differences oscillate wildly at these
jumps and can even converge to the WRONG shock speed. The FINITE-VOLUME method fixes both problems by
tracking cell AVERAGES and updating them through fluxes across cell faces:

    u_i^{n+1} = u_i^n - (dt/dx) (F_{i+1/2} - F_{i-1/2})

Because the update only moves conserved quantity between neighbours, the total is conserved exactly
(to round-off), and if the numerical flux F is chosen consistently the scheme converges to the correct
weak solution -- shock speed and all -- as guaranteed by the Lax-Wendroff theorem.

The heart is the FACE FLUX, and this module implements two ways to compute it:

  GODUNOV (first order). At each face solve the RIEMANN PROBLEM exactly -- the evolution of a single
  jump between the two neighbouring cell averages -- and take the flux at the face. For a convex flux
  this reduces to a simple min/max rule that automatically picks entropy-satisfying shocks and
  rarefactions. Robust and monotone, but it smears shocks over several cells.

  MUSCL (second order). Reconstruct a LINEAR profile inside each cell from limited slopes (the
  minmod/superbee SLOPE LIMITER suppresses spurious oscillations near jumps), evaluate the Riemann
  flux from the reconstructed face values, and advance with a two-stage SSP Runge-Kutta step. Sharp
  shocks, no overshoot, TOTAL-VARIATION-DIMINISHING.

This module solves the linear advection and inviscid Burgers equations with both schemes under
periodic boundaries, plus the exact Riemann solvers. It is validated against analytic truth:
advection transports a profile at the exact wave speed while conserving its mean and (for Godunov)
never increasing its total variation; Burgers forms a shock that travels at the Rankine-Hugoniot speed
(f(uL)-f(uR))/(uL-uR); a rarefaction fan spreads correctly; the total conserved quantity is preserved
to round-off; and MUSCL is measurably higher-order (smaller error on a smooth profile) than Godunov.
Pure stdlib; the shock-capturing companion to the Crank-Nicolson, Thomas, and Poisson PDE tools."""

from __future__ import annotations


# ---- flux functions ---------------------------------------------------------------------------

def advection_flux(a):
    """Linear advection f(u) = a u with speed a."""
    def f(u):
        return a * u
    f.speed = lambda u: a
    return f


def burgers_flux(u):
    """Inviscid Burgers f(u) = u^2 / 2."""
    return 0.5 * u * u


def _burgers_speed(u):
    return u


# ---- exact Riemann-problem fluxes --------------------------------------------------------------

def godunov_flux_advection(uL, uR, a):
    """Upwind (exact Godunov) flux for linear advection."""
    return a * uL if a >= 0 else a * uR


def godunov_flux_burgers(uL, uR):
    """Exact Godunov flux for Burgers from the Riemann solution between uL and uR."""
    if uL <= uR:
        # rarefaction (or constant): min of f over [uL, uR]
        if uL <= 0 <= uR:
            return 0.0  # sonic point, f(0)=0 is the minimum
        return min(burgers_flux(uL), burgers_flux(uR))
    else:
        # shock: speed s = (uL+uR)/2; flux from the upwind side
        s = 0.5 * (uL + uR)
        return burgers_flux(uL) if s >= 0 else burgers_flux(uR)


# ---- first-order Godunov solver ----------------------------------------------------------------

def _faces_godunov(u, equation, a):
    n = len(u)
    F = [0.0] * (n + 1)
    for i in range(n + 1):
        uL = u[(i - 1) % n]
        uR = u[i % n]
        if equation == "advection":
            F[i] = godunov_flux_advection(uL, uR, a)
        else:
            F[i] = godunov_flux_burgers(uL, uR)
    return F


def godunov_step(u, dt, dx, equation="burgers", a=1.0):
    """One first-order Godunov update with periodic boundaries."""
    n = len(u)
    F = _faces_godunov(u, equation, a)
    return [u[i] - dt / dx * (F[i + 1] - F[i]) for i in range(n)]


# ---- MUSCL (second-order) with slope limiter ---------------------------------------------------

def minmod(a, b):
    if a * b <= 0:
        return 0.0
    return a if abs(a) < abs(b) else b


def _limited_slopes(u):
    n = len(u)
    s = [0.0] * n
    for i in range(n):
        left = u[i] - u[(i - 1) % n]
        right = u[(i + 1) % n] - u[i]
        s[i] = minmod(left, right)
    return s


def _faces_muscl(u, equation, a):
    n = len(u)
    s = _limited_slopes(u)
    # reconstructed face values: at face i (between cell i-1 and i)
    F = [0.0] * (n + 1)
    for i in range(n + 1):
        il = (i - 1) % n
        ir = i % n
        uL = u[il] + 0.5 * s[il]   # right edge of left cell
        uR = u[ir] - 0.5 * s[ir]   # left edge of right cell
        if equation == "advection":
            F[i] = godunov_flux_advection(uL, uR, a)
        else:
            F[i] = godunov_flux_burgers(uL, uR)
    return F


def _muscl_rhs(u, dx, equation, a):
    n = len(u)
    F = _faces_muscl(u, equation, a)
    return [-(F[i + 1] - F[i]) / dx for i in range(n)]


def muscl_step(u, dt, dx, equation="burgers", a=1.0):
    """One second-order SSP-RK2 MUSCL update with periodic boundaries."""
    n = len(u)
    k1 = _muscl_rhs(u, dx, equation, a)
    u1 = [u[i] + dt * k1[i] for i in range(n)]
    k2 = _muscl_rhs(u1, dx, equation, a)
    return [0.5 * (u[i] + u1[i] + dt * k2[i]) for i in range(n)]


# ---- drivers -----------------------------------------------------------------------------------

def solve(u0, dx, t_final, cfl=0.4, equation="burgers", a=1.0, scheme="godunov"):
    """Evolve initial data u0 to t_final under the chosen scheme. Returns the final cell averages."""
    u = list(u0)
    t = 0.0
    step = godunov_step if scheme == "godunov" else muscl_step
    while t < t_final - 1e-15:
        # CFL-limited time step from the max wave speed
        if equation == "advection":
            smax = abs(a)
        else:
            smax = max(abs(ui) for ui in u) or 1e-12
        dt = cfl * dx / smax
        if t + dt > t_final:
            dt = t_final - t
        u = step(u, dt, dx, equation, a)
        t += dt
    return u


def total_mass(u, dx):
    return sum(u) * dx


def total_variation(u):
    """Sum of |u_{i+1} - u_i| over periodic cells -- a TVD scheme never increases this."""
    n = len(u)
    return sum(abs(u[(i + 1) % n] - u[i]) for i in range(n))


def rankine_hugoniot_speed(uL, uR, flux=burgers_flux):
    """Shock speed s = (f(uL) - f(uR)) / (uL - uR)."""
    if uL == uR:
        return _burgers_speed(uL)
    return (flux(uL) - flux(uR)) / (uL - uR)
