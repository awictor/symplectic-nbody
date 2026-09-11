"""Tests for galton.py -- the Galton board (binomial -> Gaussian).

Self-running: prints PASS/FAIL per check, exits 1 if any fail. The binomial slot distribution
is checked against exact values, the CLT normal limit, and a seeded Monte-Carlo bead drop.
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import galton as g  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


def approx(a, b, tol):
    return abs(a - b) <= tol


# --- binomial slot probabilities -------------------------------------------
sp = g.slot_probabilities(10)
check("there are rows+1 slots", len(sp) == 11)
check("slot probabilities sum to 1", approx(sum(sp), 1.0, 1e-12))
check("all slot probabilities nonnegative", all(x >= 0 for x in sp))
# symmetric for p = 1/2
check("fair board is symmetric", all(approx(sp[k], sp[10 - k], 1e-12) for k in range(11)))
# exact known value: C(10,5)/2^10 = 252/1024
check("center slot is C(10,5)/2^10", approx(sp[5], 252 / 1024, 1e-12))
# a single row is a fair coin: 1/2, 1/2
check("one row is a coin flip", approx(g.slot_probabilities(1)[0], 0.5, 1e-12))
# zero rows: one slot with probability 1
check("zero rows lands in one slot", g.slot_probabilities(0) == [1.0])
try:
    g.slot_probabilities(10, 1.5)
    check("rejects p out of range", False)
except ValueError:
    check("rejects p out of range", True)

# --- moments ----------------------------------------------------------------
check("mean is n p", approx(g.mean_slot(10, 0.5), 5.0, 1e-12))
check("variance is n p (1-p)", approx(g.variance_slot(10, 0.5), 2.5, 1e-12))
check("mode of a fair 10-row board is 5", g.mode_slot(10, 0.5) == 5)
# moments match the distribution directly
mean_direct = sum(k * sp[k] for k in range(11))
var_direct = sum((k - mean_direct) ** 2 * sp[k] for k in range(11))
check("distribution mean matches n p", approx(mean_direct, g.mean_slot(10), 1e-9))
check("distribution variance matches n p(1-p)", approx(var_direct, g.variance_slot(10), 1e-9))
# biased board shifts the peak toward np
check("biased board mean shifts to n p", approx(g.mean_slot(10, 0.7), 7.0, 1e-12))
sp7 = g.slot_probabilities(10, 0.7)
check("biased board peaks near n p", sp7.index(max(sp7)) in (6, 7))

# --- normal (CLT) approximation --------------------------------------------
na = g.normal_approx(50)
mu, sigma = g.mean_slot(50), math.sqrt(g.variance_slot(50))
check("normal density peaks at the mean", na.index(max(na)) == int(round(mu)))
check("normal pdf integrates (sums) to ~1 over the slots", approx(sum(na), 1.0, 0.02))
# de Moivre-Laplace: peak height ~ 1/(sigma sqrt(2 pi))
check("normal peak height ~ 1/(sigma sqrt(2 pi))",
      approx(max(na), 1.0 / (sigma * math.sqrt(2 * math.pi)), 1e-9))
# the normal approximation should be close to the binomial for large n
binom50 = g.slot_probabilities(50)
check("normal approx close to binomial at the center", approx(na[25], binom50[25], 5e-3))

# --- convergence: TVD shrinks with n ---------------------------------------
tv10 = g.total_variation_distance(10)
tv50 = g.total_variation_distance(50)
tv200 = g.total_variation_distance(200)
check("total-variation distance is small", tv10 < 0.05)
check("TVD decreases as rows grow", tv10 > tv50 > tv200)
# CLT rate: TVD ~ 1/sqrt(n), so doubling-and-then-some should roughly track sqrt
check("TVD roughly follows the 1/sqrt(n) rate",
      approx(tv50 / tv10, math.sqrt(10 / 50), 0.3))

# --- Monte-Carlo agreement --------------------------------------------------
sim = g.simulate(16, beads=40000, p=0.5, seed=5)
binom16 = g.slot_probabilities(16)
check("simulated histogram sums to 1", approx(sum(sim), 1.0, 1e-9))
check("simulated histogram matches the binomial", all(approx(sim[k], binom16[k], 0.01) for k in range(17)))
check("simulated mean matches n p",
      approx(sum(k * sim[k] for k in range(17)), g.mean_slot(16), 0.1))
# biased simulation shifts the peak
simb = g.simulate(10, beads=40000, p=0.7, seed=3)
check("biased simulation peaks near n p", simb.index(max(simb)) in (6, 7))


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall galton tests passed")
