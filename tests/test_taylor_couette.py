"""Validate Taylor-Couette: exact profile, boundary conditions, Rayleigh criterion, Taylor onset."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import taylor_couette as tc


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Taylor-Couette flow tests")

    r1, r2 = 1.0, 2.0
    o1, o2 = 3.0, 1.0

    # --- profile satisfies both no-slip boundary conditions ---
    check(f"Omega(r1) == omega1 ({tc.angular_velocity(r1, r1, r2, o1, o2):.6f})",
          abs(tc.angular_velocity(r1, r1, r2, o1, o2) - o1) < 1e-12)
    check(f"Omega(r2) == omega2 ({tc.angular_velocity(r2, r1, r2, o1, o2):.6f})",
          abs(tc.angular_velocity(r2, r1, r2, o1, o2) - o2) < 1e-12)
    check("v(r1) == omega1*r1", abs(tc.velocity(r1, r1, r2, o1, o2) - o1 * r1) < 1e-12)
    check("v(r2) == omega2*r2", abs(tc.velocity(r2, r1, r2, o1, o2) - o2 * r2) < 1e-12)

    # --- Omega(r) = A + B/r^2 satisfies the governing ODE d/dr[ (1/r) d(r^2 Omega)/dr ] = 0 ---
    # equivalently L(r) = A r^2 + B is a linear function of r^2, so its second derivative wrt r^2 is 0.
    a, b = tc.couette_coeffs(r1, r2, o1, o2)
    # sample: L(r) - (a r^2 + b) == 0
    ok = True
    for i in range(11):
        r = r1 + (r2 - r1) * i / 10
        if abs(tc.specific_angular_momentum(r, r1, r2, o1, o2) - (a * r * r + b)) > 1e-12:
            ok = False
    check("L(r) == A r^2 + B exactly", ok)

    # --- solid-body rotation: omega1 == omega2 -> B == 0, rigid rotation, stable ---
    a2, b2 = tc.couette_coeffs(r1, r2, 2.0, 2.0)
    check(f"solid body -> B==0 ({b2:.2e}) and A==omega", abs(b2) < 1e-12 and abs(a2 - 2.0) < 1e-12)
    check("solid body is Rayleigh stable", tc.is_rayleigh_stable(r1, r2, 2.0, 2.0))

    # --- Rayleigh discriminant Phi = 4 A Omega(r): cross-check against numerical d(L^2)/dr / r^3 ---
    def num_phi(r):
        h = 1e-6
        l_plus = tc.specific_angular_momentum(r + h, r1, r2, o1, o2) ** 2
        l_minus = tc.specific_angular_momentum(r - h, r1, r2, o1, o2) ** 2
        return (l_plus - l_minus) / (2 * h) / r ** 3
    r = 1.5
    check(f"Phi matches numerical d(L^2)/dr/r^3 at r=1.5",
          abs(tc.rayleigh_discriminant(r, r1, r2, o1, o2) - num_phi(r)) < 1e-4)

    # --- inner spins, outer at rest: L^2 decreases outward -> Rayleigh UNSTABLE ---
    check("inner-only rotation is Rayleigh UNSTABLE", not tc.is_rayleigh_stable(r1, r2, 5.0, 0.0))
    l_in = tc.specific_angular_momentum(r1, r1, r2, 5.0, 0.0) ** 2
    l_out = tc.specific_angular_momentum(r2, r1, r2, 5.0, 0.0) ** 2
    check(f"L^2 decreases outward for inner-only ({l_in:.2f} > {l_out:.2f})", l_in > l_out)

    # --- outer spinning faster (co-rotation above the line): stable ---
    check("fast outer co-rotation is Rayleigh stable", tc.is_rayleigh_stable(r1, r2, 1.0, 3.0))

    # --- Rayleigh marginal line: Omega2/Omega1 == (r1/r2)^2 gives constant L (potential vortex) ---
    ratio = tc.rayleigh_marginal_ratio(r1, r2)
    o1m, o2m = 4.0, 4.0 * ratio
    am, bm = tc.couette_coeffs(r1, r2, o1m, o2m)
    check(f"marginal line -> A ~ 0 (constant L) ({am:.2e})", abs(am) < 1e-9)
    # on the marginal line Phi = 4 A Omega ~ 0 everywhere
    check("marginal line -> Phi ~ 0 (irrotational vortex)",
          abs(tc.rayleigh_discriminant(1.5, r1, r2, o1m, o2m)) < 1e-8)

    # --- torque is independent of radius (uniform angular-momentum transport) ---
    # G = 4 pi mu B; also stress*2 pi r^2 should be constant -- check via B directly and via two radii
    g = tc.torque_per_length(r1, r2, o1, o2, mu=2.5)
    check(f"torque per length finite and set by B ({g:.4f})", abs(g - (-4 * math.pi * 2.5 * b)) < 1e-12)

    # --- Taylor number scaling and onset ---
    # Ta = Omega1^2 r1 (r2-r1)^3 / nu^2
    ta = tc.taylor_number(1.0, 1.1, 100.0, 1.0)  # d=0.1
    check(f"Taylor number matches formula ({ta:.4f})",
          abs(ta - (100.0 ** 2 * 1.0 * 0.1 ** 3 / 1.0)) < 1e-9)
    # below critical: slow inner cylinder -> stable Couette survives
    check("small Ta -> stable (no vortices)", not tc.is_taylor_unstable(1.0, 1.1, 1.0, 1.0))
    # above critical: fast inner cylinder -> Taylor vortices (Ta = 2000^2 * 0.001 = 4000 > 1708)
    check("large Ta -> Taylor vortices", tc.is_taylor_unstable(1.0, 1.1, 2000.0, 1.0))
    # Ta grows as Omega^2
    check("Ta ~ Omega^2", abs(tc.taylor_number(1, 1.1, 2, 1) / tc.taylor_number(1, 1.1, 1, 1) - 4.0) < 1e-9)

    # --- bad geometry rejected ---
    try:
        tc.couette_coeffs(2.0, 1.0, 1.0, 1.0)
        check("rejects r1 >= r2", False)
    except ValueError:
        check("rejects r1 >= r2", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
