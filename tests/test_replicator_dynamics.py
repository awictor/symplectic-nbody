"""Validate replicator dynamics: simplex preservation, dominant fixation, Nash mixed eq, RPS conservation."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import replicator_dynamics as rd


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Replicator dynamics tests")

    # --- simplex preserved: sum 1, non-negative, all along a trajectory ---
    A = [[1, 2, 0], [0, 1, 2], [2, 0, 1]]  # rock-paper-scissors-ish
    times, traj = rd.integrate(A, [0.2, 0.5, 0.3], t_max=20.0, dt=0.01, record_every=50)
    ok_sum = all(abs(sum(x) - 1.0) < 1e-9 for x in traj)
    ok_nonneg = all(all(xi >= -1e-12 for xi in x) for x in traj)
    check("frequencies stay on the simplex (sum 1)", ok_sum)
    check("frequencies non-negative", ok_nonneg)

    # --- strictly dominant strategy sweeps to fixation ---
    # strategy 0 dominates: it earns more against every opponent
    dom = [[3, 3, 3], [1, 2, 1], [0, 1, 2]]
    times, traj = rd.integrate(dom, [1 / 3, 1 / 3, 1 / 3], t_max=50.0, dt=0.01, record_every=100)
    final = traj[-1]
    check(f"dominant strategy fixes ({final[0]:.3f})", final[0] > 0.99)

    # --- two-strategy game converges to the interior Nash mixed equilibrium ---
    # hawk-dove-like: A = [[0, 3], [1, 2]]; interior eq where payoffs equal
    # f1 = 0*x1 + 3*x2 = 3 x2 ; f2 = 1*x1 + 2*x2. Equal: 3 x2 = x1 + 2 x2 -> x2 = x1 -> x1=x2=0.5
    hd = [[0, 3], [1, 2]]
    times, traj = rd.integrate(hd, [0.9, 0.1], t_max=60.0, dt=0.01, record_every=100)
    final = traj[-1]
    check(f"converges to interior Nash (0.5, 0.5) (got {final[0]:.3f})", abs(final[0] - 0.5) < 0.02)

    # --- from a different start, same interior equilibrium ---
    times, traj = rd.integrate(hd, [0.1, 0.9], t_max=60.0, dt=0.01, record_every=100)
    check("interior eq is a global attractor here", abs(traj[-1][0] - 0.5) < 0.02)

    # --- rock-paper-scissors: orbits and conserves the product x1 x2 x3 ---
    rps = [[0, -1, 1], [1, 0, -1], [-1, 1, 0]]
    x0 = [0.4, 0.35, 0.25]
    p0 = rd.rps_conserved(x0)
    times, traj = rd.integrate(rps, x0, t_max=30.0, dt=0.005, record_every=100)
    products = [rd.rps_conserved(x) for x in traj]
    # the product is conserved (small drift from RK4)
    check(f"RPS conserves product ({min(products):.4f}..{max(products):.4f})",
          max(products) - min(products) < 0.02 * p0)
    # and it orbits (doesn't settle) -- the frequency of strategy 1 varies a lot
    x1s = [x[0] for x in traj]
    check("RPS orbits (does not fix)", max(x1s) - min(x1s) > 0.05)

    # --- interior equilibrium (1/3,1/3,1/3) is a rest point for RPS ---
    check("RPS center is a rest point", rd.is_rest_point(rps, [1 / 3, 1 / 3, 1 / 3]))

    # --- pure strategies are rest points ---
    check("pure strategy is a rest point", rd.is_rest_point(A, [1.0, 0.0, 0.0]))
    check("another pure strategy is a rest point", rd.is_rest_point(A, [0.0, 1.0, 0.0]))

    # --- fitness and mean fitness ---
    x = [0.5, 0.3, 0.2]
    f = rd.fitness(A, x)
    check("fitness = A x", f == [A[i][0] * 0.5 + A[i][1] * 0.3 + A[i][2] * 0.2 for i in range(3)])
    check("mean fitness = x.Ax", abs(rd.mean_fitness(A, x) - sum(x[i] * f[i] for i in range(3))) < 1e-12)

    # --- velocity zero at a rest point ---
    v = rd.velocity(rps, [1 / 3, 1 / 3, 1 / 3])
    check("velocity ~ 0 at RPS center", all(abs(vi) < 1e-9 for vi in v))

    # --- mean fitness non-decreasing for a symmetric (partnership) game ---
    # symmetric A: replicator increases mean fitness (Fisher's theorem analogue)
    sym = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    times, traj = rd.integrate(sym, [0.5, 0.3, 0.2], t_max=20.0, dt=0.01, record_every=50)
    phis = [rd.mean_fitness(sym, x) for x in traj]
    check("mean fitness non-decreasing (symmetric game)",
          all(phis[i] <= phis[i + 1] + 1e-6 for i in range(len(phis) - 1)))

    # --- deterministic ---
    t1, tr1 = rd.integrate(A, [0.2, 0.5, 0.3], t_max=5.0, dt=0.01, record_every=100)
    t2, tr2 = rd.integrate(A, [0.2, 0.5, 0.3], t_max=5.0, dt=0.01, record_every=100)
    check("deterministic", tr1 == tr2)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
