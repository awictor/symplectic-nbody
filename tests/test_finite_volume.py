"""Tests for finite volume: advection transport, mass conservation, shock speed, TVD, MUSCL order."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finite_volume import (  # noqa: E402
    solve,
    godunov_step,
    muscl_step,
    total_mass,
    total_variation,
    rankine_hugoniot_speed,
    burgers_flux,
    godunov_flux_burgers,
    minmod,
)


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    # ---- 1. linear advection transports a profile at the right speed --------------------
    n = 200
    L = 1.0
    dx = L / n
    a = 1.0
    # a smooth bump
    u0 = [math.sin(2 * math.pi * i * dx) for i in range(n)]
    t = 0.3
    u = solve(u0, dx, t, cfl=0.4, equation="advection", a=a, scheme="godunov")
    # exact: u(x, t) = sin(2 pi (x - a t))
    exact = [math.sin(2 * math.pi * (i * dx - a * t)) for i in range(n)]
    err = max(abs(u[i] - exact[i]) for i in range(n))
    check("advection transports at wave speed (Godunov)", err < 0.15, f"max err {err:.3f}")

    # ---- 2. MUSCL advection is more accurate than Godunov -------------------------------
    u_g = solve(u0, dx, t, cfl=0.4, equation="advection", a=a, scheme="godunov")
    u_m = solve(u0, dx, t, cfl=0.4, equation="advection", a=a, scheme="muscl")
    err_g = max(abs(u_g[i] - exact[i]) for i in range(n))
    err_m = max(abs(u_m[i] - exact[i]) for i in range(n))
    check("MUSCL more accurate than Godunov", err_m < err_g, f"MUSCL {err_m:.3f} vs Godunov {err_g:.3f}")

    # ---- 3. mass conserved to round-off (periodic) --------------------------------------
    m0 = total_mass(u0, dx)
    check("advection conserves mass", abs(total_mass(u_g, dx) - m0) < 1e-9,
          f"{total_mass(u_g, dx)} vs {m0}")

    # ---- 4. Godunov is TVD for advection (total variation not increasing) ---------------
    # step a step function; TV should not grow
    step_ic = [1.0 if i < n // 2 else 0.0 for i in range(n)]
    tv0 = total_variation(step_ic)
    u = list(step_ic)
    for _ in range(50):
        u = godunov_step(u, 0.4 * dx / abs(a), dx, "advection", a)
    check("Godunov advection is TVD", total_variation(u) <= tv0 + 1e-9,
          f"TV {total_variation(u):.4f} vs {tv0:.4f}")

    # ---- 5. Burgers shock travels at the Rankine-Hugoniot speed -------------------------
    # Riemann problem uL=1 (left half), uR=0 (right half) -> shock speed (f(1)-f(0))/(1-0)=0.5
    n = 400
    dx = 1.0 / n
    uL, uR = 1.0, 0.0
    ic = [uL if i < n // 4 else uR for i in range(n)]
    x0 = (n // 4) * dx  # initial shock position
    s = rankine_hugoniot_speed(uL, uR)
    check("Rankine-Hugoniot speed = 0.5", abs(s - 0.5) < 1e-12, f"{s}")
    t = 0.2
    u = solve(ic, dx, t, cfl=0.4, equation="burgers", scheme="godunov")
    # find the shock location (midpoint crossing of 0.5)
    shock_pos = None
    for i in range(n - 1):
        if u[i] >= 0.5 > u[i + 1]:
            shock_pos = (i + 0.5) * dx
            break
    expected = x0 + s * t
    check("Burgers shock at RH position", shock_pos is not None and abs(shock_pos - expected) < 3 * dx,
          f"shock {shock_pos} vs expected {expected:.4f}")

    # ---- 6. Burgers conserves mass ------------------------------------------------------
    check("Burgers conserves mass", abs(total_mass(u, dx) - total_mass(ic, dx)) < 1e-9)

    # ---- 7. Burgers no new maxima (bounded by initial range) ----------------------------
    check("Burgers stays in [uR, uL]", all(uR - 1e-9 <= v <= uL + 1e-9 for v in u))

    # ---- 8. rarefaction: uL=0, uR=1 spreads (no shock, smooth fan) ----------------------
    ic = [0.0 if i < n // 2 else 1.0 for i in range(n)]
    u = solve(ic, dx, 0.2, cfl=0.4, equation="burgers", scheme="godunov")
    # the jump should have spread: count cells strictly between 0.05 and 0.95
    transition = sum(1 for v in u if 0.05 < v < 0.95)
    check("rarefaction spreads the jump", transition > 5, f"{transition} transition cells")
    check("rarefaction conserves mass", abs(total_mass(u, dx) - total_mass(ic, dx)) < 1e-9)

    # ---- 9. exact Godunov Burgers flux values -------------------------------------------
    check("godunov flux shock uL>uR", abs(godunov_flux_burgers(1.0, 0.0) - 0.5) < 1e-12)
    check("godunov flux sonic rarefaction", abs(godunov_flux_burgers(-1.0, 1.0) - 0.0) < 1e-12)
    check("godunov flux constant", abs(godunov_flux_burgers(2.0, 2.0) - burgers_flux(2.0)) < 1e-12)

    # ---- 10. minmod limiter properties --------------------------------------------------
    check("minmod opposite signs -> 0", minmod(1.0, -2.0) == 0.0)
    check("minmod picks smaller magnitude", minmod(3.0, 1.0) == 1.0 and minmod(-1.0, -5.0) == -1.0)

    # ---- 11. MUSCL is TVD-ish on a step (no large overshoot) ----------------------------
    ic = [1.0 if i < n // 2 else 0.0 for i in range(n)]
    u = solve(ic, dx, 0.15, cfl=0.4, equation="burgers", scheme="muscl")
    check("MUSCL no overshoot", max(u) <= 1.0 + 1e-6 and min(u) >= -1e-6, f"range [{min(u):.4f},{max(u):.4f}]")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
