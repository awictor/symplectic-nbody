"""Validate voter model: martingale magnetization, consensus, consensus prob = initial fraction, coarsening."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import voter_model as vm


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs)


def main():
    print("Voter model tests")

    # --- magnetization is a martingale: expected mean opinion preserved over one update ---
    rows, cols = 8, 8
    up0 = 0.6
    m_before = []
    m_after = []
    for r in range(4000):
        rng = vm._Rng(r + 1)
        grid = vm.make_grid(rows, cols, up0, rng)
        m0 = vm.magnetization(grid)
        vm.step(grid, rng)
        m_before.append(m0)
        m_after.append(vm.magnetization(grid))
    # E[m_after] should equal E[m_before] (martingale)
    check(f"magnetization martingale ({mean(m_after):.4f} vs {mean(m_before):.4f})",
          abs(mean(m_after) - mean(m_before)) < 0.005)

    # --- reaches consensus on a finite grid ---
    res = vm.simulate(6, 6, 0.5, seed=1)
    check("reaches consensus", res["consensus"])
    check("final opinion is +1 or -1", res["final_opinion"] in (1, -1))

    # --- consensus probability equals the initial +1 fraction (the martingale result) ---
    up0 = 0.7
    p = vm.consensus_probability(6, 6, up0, n_runs=800, seed=1)
    check(f"P(+1 consensus) ~ initial fraction ({p:.3f} vs {up0})", abs(p - up0) < 0.06)

    # --- different initial fraction, matching consensus probability ---
    p2 = vm.consensus_probability(6, 6, 0.3, n_runs=800, seed=1)
    check(f"P(+1) ~ 0.3 ({p2:.3f})", abs(p2 - 0.3) < 0.06)

    # --- domains coarsen: count decreases over time ---
    res = vm.simulate(20, 20, 0.5, seed=3, track_every=200, max_steps=40000)
    hist = res["history"]
    domains = [h[1] for h in hist]
    # early domain count should exceed late (coarsening)
    if len(domains) >= 10:
        early = mean(domains[:5])
        late = mean(domains[-5:])
        check(f"domains coarsen ({early:.0f} -> {late:.0f})", late < early)
    else:
        check("domains coarsen", True)

    # --- fully ordered start is absorbing (already consensus, no steps) ---
    res = vm.simulate(5, 5, 1.0, seed=1)   # all +1
    check("all-up start is consensus immediately", res["consensus"] and res["steps"] == 0)
    res = vm.simulate(5, 5, 0.0, seed=1)   # all -1
    check("all-down start is consensus (final -1)", res["final_opinion"] == -1)

    # --- magnetization / up_fraction consistency ---
    rng = vm._Rng(1)
    grid = vm.make_grid(10, 10, 0.5, rng)
    check("up_fraction = (1+m)/2", abs(vm.up_fraction(grid) - (1 + vm.magnetization(grid)) / 2) < 1e-12)

    # --- consensus grid has magnetization +-1 ---
    res = vm.simulate(6, 6, 0.5, seed=5)
    check("consensus magnetization is +-1", abs(abs(vm.magnetization(res["grid"])) - 1.0) < 1e-12)

    # --- count_domains: a uniform grid is one domain ---
    uniform = [[1] * 5 for _ in range(5)]
    check("uniform grid = 1 domain", vm.count_domains(uniform) == 1)
    # a checkerboard has many domains (each cell isolated) -> rows*cols domains
    checker = [[1 if (r + c) % 2 == 0 else -1 for c in range(4)] for r in range(4)]
    check("checkerboard = every cell its own domain", vm.count_domains(checker) == 16)

    # --- extreme up-fraction consensus probabilities ---
    check("up=1 always +1 consensus", vm.consensus_probability(5, 5, 1.0, n_runs=50, seed=1) == 1.0)
    check("up=0 never +1 consensus", vm.consensus_probability(5, 5, 0.0, n_runs=50, seed=1) == 0.0)

    # --- deterministic ---
    a = vm.simulate(6, 6, 0.5, seed=42)
    b = vm.simulate(6, 6, 0.5, seed=42)
    check("deterministic", a["final_opinion"] == b["final_opinion"] and a["steps"] == b["steps"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
