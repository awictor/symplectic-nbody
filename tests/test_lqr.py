"""Tests for lqr: Riccati residual, closed-loop stability, scalar closed form, optimality."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lqr import (solve_dare, lqr_gain, finite_horizon_gains, closed_loop_matrix,  # noqa: E402
                 simulate, lqr_cost, spectral_radius_2x2, _mm, _T, _add, _sub, _inverse)


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def riccati_residual(A, B, Q, R, P):
    """Max |P - (Q + A^T P A - A^T P B (R+B^T P B)^-1 B^T P A)|."""
    At, Bt = _T(A), _T(B)
    BtP = _mm(Bt, P)
    S = _add(R, _mm(BtP, B))
    AtP = _mm(At, P)
    rhs = _add(_sub(_mm(AtP, A), _mm(_mm(_mm(AtP, B), _inverse(S)), _mm(BtP, A))), Q)
    return max(abs(P[i][j] - rhs[i][j]) for i in range(len(P)) for j in range(len(P[0])))


def main():
    # ---- 1. double integrator (cart): P satisfies Riccati, closed loop stable ---------
    dt = 0.1
    A = [[1.0, dt], [0.0, 1.0]]
    B = [[0.5 * dt * dt], [dt]]
    Q = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.1]]
    K, P = lqr_gain(A, B, Q, R)
    check("Riccati residual ~ 0 (double integrator)", riccati_residual(A, B, Q, R, P) < 1e-8,
          f"{riccati_residual(A, B, Q, R, P):.2e}")
    Acl = closed_loop_matrix(A, B, K)
    sr = spectral_radius_2x2(Acl)
    check("closed loop stable (spectral radius < 1)", sr < 1.0, f"rho={sr:.4f}")

    # simulate from a displaced state -> decays to origin
    xs = simulate(A, B, K, [5.0, 0.0], 200)
    final = math.hypot(xs[-1][0], xs[-1][1])
    check("closed loop drives state to origin", final < 1e-3, f"final |x|={final:.2e}")

    # ---- 2. scalar system matches the closed-form Riccati -----------------------------
    # x+ = a x + b u, cost q x^2 + r u^2. DARE: p = q + a^2 p - a^2 p^2 b^2/(r + p b^2)
    a, b, q, r = 1.2, 1.0, 2.0, 1.0
    K1, P1 = lqr_gain([[a]], [[b]], [[q]], [[r]])
    p = P1[0][0]
    # closed-form: solve the quadratic p = q + a^2 p r / (r + p b^2)
    # -> b^2 p^2 - (b^2 q + a^2 r - r) p ... solve directly by the fixed point already; verify residual
    resid = abs(p - (q + a * a * p - a * a * p * p * b * b / (r + p * b * b)))
    check("scalar Riccati residual ~ 0", resid < 1e-10, f"{resid:.2e}")
    # gain closed form K = b p a / (r + b^2 p)
    k_closed = b * p * a / (r + b * b * p)
    check("scalar gain matches closed form", abs(K1[0][0] - k_closed) < 1e-10,
          f"{K1[0][0]} vs {k_closed}")
    # scalar closed loop stable
    check("scalar closed loop stable", abs(a - b * K1[0][0]) < 1.0, f"|a-bK|={abs(a-b*K1[0][0]):.4f}")

    # ---- 3. optimality: LQR cost <= randomly perturbed stabilizing gains --------------
    x0 = [3.0, -1.0]
    base_cost = lqr_cost(A, B, K, Q, R, x0, 300)
    rng = LCG(7)
    worse = 0
    for _ in range(100):
        Kp = [[K[0][j] + 0.3 * (rng.u() - 0.5) for j in range(2)]]
        # only compare against stabilizing gains
        if spectral_radius_2x2(closed_loop_matrix(A, B, Kp)) < 1.0:
            if lqr_cost(A, B, Kp, Q, R, x0, 300) < base_cost - 1e-9:
                worse += 1
    check("LQR gain is cost-optimal (no perturbed gain beats it)", worse == 0,
          f"{worse} perturbed gains cheaper")

    # ---- 4. heavier R -> gentler gain; heavier Q -> stronger gain ---------------------
    K_cheap, _ = lqr_gain(A, B, Q, [[0.01]])    # cheap control -> aggressive
    K_exp, _ = lqr_gain(A, B, Q, [[10.0]])      # expensive control -> gentle
    norm = lambda M: math.sqrt(sum(M[0][j] ** 2 for j in range(len(M[0]))))
    check("heavier control penalty R -> smaller gain", norm(K_exp) < norm(K_cheap),
          f"|K_exp|={norm(K_exp):.3f} |K_cheap|={norm(K_cheap):.3f}")
    K_bigQ, _ = lqr_gain(A, B, [[100.0, 0], [0, 100.0]], R)
    check("heavier state penalty Q -> larger gain", norm(K_bigQ) > norm(K), f"{norm(K_bigQ):.3f}")

    # ---- 5. finite horizon converges to infinite horizon ------------------------------
    gains = finite_horizon_gains(A, B, Q, R, 200)
    K0 = gains[0]     # earliest gain, far from the terminal -> should match infinite horizon
    diff = max(abs(K0[0][j] - K[0][j]) for j in range(2))
    check("finite-horizon gain converges to infinite-horizon", diff < 1e-6, f"diff {diff:.2e}")

    # ---- 6. a naturally unstable system gets stabilized -------------------------------
    Au = [[1.5, 1.0], [0.0, 1.2]]     # both eigenvalues > 1 (unstable)
    Bu = [[0.0], [1.0]]
    Ku, Pu = lqr_gain(Au, Bu, [[1.0, 0], [0, 1.0]], [[1.0]])
    check("unstable system: open loop unstable", spectral_radius_2x2(Au) > 1.0)
    check("unstable system: LQR stabilizes it",
          spectral_radius_2x2(closed_loop_matrix(Au, Bu, Ku)) < 1.0,
          f"rho_cl={spectral_radius_2x2(closed_loop_matrix(Au, Bu, Ku)):.4f}")
    xs = simulate(Au, Bu, Ku, [1.0, 1.0], 300)
    check("unstable system decays under LQR", math.hypot(*xs[-1]) < 1e-2, f"{math.hypot(*xs[-1]):.2e}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
