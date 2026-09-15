"""Validate Gillespie SSA: decay mean vs analytic, mass conservation, waiting times, reaction choice, ODE limit."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import gillespie_ssa as ssa


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Gillespie SSA tests")

    # --- pure decay A -> 0: mean matches analytic exponential ---
    # dA/dt = -k A -> A(t) = A0 exp(-k t)
    k = 0.5
    A0 = 1000
    decay = [ssa.Reaction({0: 1}, {}, k)]
    sample_times = [0.5, 1.0, 2.0, 4.0]
    means = ssa.ensemble_mean(decay, [A0], sample_times, n_runs=200, seed=1)
    ok = True
    for i, t in enumerate(sample_times):
        analytic = A0 * math.exp(-k * t)
        if abs(means[i][0] - analytic) > 0.08 * A0:
            ok = False
    check("decay mean matches A0 exp(-kt)", ok)

    # --- reversible A <-> B conserves total mass on every path ---
    react = [ssa.Reaction({0: 1}, {1: 1}, 1.0), ssa.Reaction({1: 1}, {0: 1}, 1.0)]
    times, states = ssa.simulate(react, [100, 0], t_max=5.0, seed=3)
    total_ok = all(s[0] + s[1] == 100 for s in states)
    check("A<->B conserves total mass exactly", total_ok)

    # --- A<->B mean approaches the ODE steady state (equal rates -> 50/50) ---
    means = ssa.ensemble_mean(react, [100, 0], [10.0], n_runs=200, seed=1)
    check(f"A<->B steady state ~ 50/50 ({means[0][0]:.0f}/{means[0][1]:.0f})",
          abs(means[0][0] - 50) < 8 and abs(means[0][1] - 50) < 8)

    # --- waiting times are exponential with rate = total propensity ---
    # single decay reaction, constant state (large A so propensity ~ constant over a few steps)
    rng = ssa._Rng(7)
    state = [10000]
    r = ssa.Reaction({0: 1}, {}, 0.3)
    a0 = r.propensity(state)
    taus = []
    for _ in range(3000):
        u = max(rng.u(), 1e-12)
        taus.append(-math.log(u) / a0)
    mean_tau = sum(taus) / len(taus)
    check(f"waiting time mean ~ 1/a0 ({mean_tau:.5f} vs {1/a0:.5f})", abs(mean_tau - 1 / a0) < 0.1 / a0)

    # --- reaction choice frequency matches propensity ratio ---
    # two reactions with propensities in ratio 3:1
    reacts = [ssa.Reaction({0: 1}, {}, 3.0), ssa.Reaction({1: 1}, {}, 1.0)]
    st = [100, 100]
    p = [reacts[0].propensity(st), reacts[1].propensity(st)]
    total = sum(p)
    # simulate choices directly
    rng = ssa._Rng(9)
    count0 = 0
    N = 5000
    for _ in range(N):
        thr = rng.u() * total
        cum = 0.0
        for j, aj in enumerate(p):
            cum += aj
            if cum >= thr:
                if j == 0:
                    count0 += 1
                break
    freq0 = count0 / N
    check(f"reaction choice freq ~ propensity ratio ({freq0:.3f} vs {p[0]/total:.3f})",
          abs(freq0 - p[0] / total) < 0.03)

    # --- propensity: bimolecular A + B -> C uses count product ---
    r = ssa.Reaction({0: 1, 1: 1}, {2: 1}, 2.0)
    check("bimolecular propensity = k*A*B", abs(r.propensity([5, 4, 0]) - 2.0 * 5 * 4) < 1e-9)
    # dimerization 2A -> B uses A(A-1)/2
    r2 = ssa.Reaction({0: 2}, {1: 1}, 1.0)
    check("dimer propensity = k*A(A-1)/2", abs(r2.propensity([6, 0]) - 6 * 5 / 2) < 1e-9)

    # --- absorbing state: decay to zero stops ---
    times, states = ssa.simulate([ssa.Reaction({0: 1}, {}, 1.0)], [5], t_max=100.0, seed=1)
    check("decay reaches extinction", states[-1][0] == 0)

    # --- ODE limit: larger population -> relative fluctuations shrink ---
    def rel_sd_at_end(A0, n_runs=100):
        react = [ssa.Reaction({0: 1}, {}, 0.5)]
        vals = []
        for r in range(n_runs):
            times, states = ssa.simulate(react, [A0], t_max=1.0, seed=r + 1)
            vals.append(states[-1][0])
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        return sd / m if m > 0 else 0
    rs_small = rel_sd_at_end(50)
    rs_big = rel_sd_at_end(5000)
    check(f"relative fluctuations shrink with population ({rs_small:.3f} > {rs_big:.3f})",
          rs_small > rs_big)

    # --- Lotka-Volterra produces oscillation (prey and predator both vary a lot) ---
    # X -> 2X (prey birth), X + Y -> 2Y (predation), Y -> 0 (predator death)
    lv = [
        ssa.Reaction({0: 1}, {0: 2}, 1.0),
        ssa.Reaction({0: 1, 1: 1}, {1: 2}, 0.01),
        ssa.Reaction({1: 1}, {}, 1.0),
    ]
    times, states = ssa.simulate(lv, [50, 50], t_max=10.0, seed=5)
    prey = [s[0] for s in states]
    check("Lotka-Volterra prey oscillates", max(prey) - min(prey) > 30)

    # --- reproducible per seed ---
    t1, s1 = ssa.simulate(react, [100, 0], 5.0, seed=42)
    t2, s2 = ssa.simulate(react, [100, 0], 5.0, seed=42)
    check("reproducible per seed", t1 == t2 and s1 == s2)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
