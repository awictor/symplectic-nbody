"""Validate Bak-Sneppen: self-organized threshold ~2/3, mean fitness rises, avalanche heavy tail, 3-site update."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import bak_sneppen as bs


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Bak-Sneppen tests")

    # --- each step replaces exactly 3 adjacent species ---
    rng = bs._Rng(1)
    fit = [rng.u() for _ in range(20)]
    before = list(fit)
    imin = bs.step(fit, rng)
    changed = [i for i in range(20) if fit[i] != before[i]]
    # the three replaced are imin-1, imin, imin+1 (mod n); some may coincidentally equal old (unlikely)
    expected = {(imin - 1) % 20, imin % 20, (imin + 1) % 20}
    check("step targets 3 adjacent sites", set(changed) <= expected and len(changed) >= 2)
    check("min index was the least fit", before[imin] == min(before))

    # --- self-organized threshold (the gap = running max of min fitness) approaches 1-D f_c ~2/3 ---
    res = bs.simulate(100, 200000, seed=1)
    check(f"threshold ~ 2/3 ({res['threshold']:.3f})", abs(res["threshold"] - 0.667) < 0.06)

    # --- the gap (running max of min fitness) rises from ~0 and plateaus at f_c ---
    res = bs.simulate(100, 100000, seed=1, track=True)
    hist = res["min_history"]
    # running max over the first 50 updates vs the whole run (the gap climbs fast then plateaus at f_c)
    gap_early = max(hist[:50])
    gap_late = max(hist)
    check(f"gap rises (self-organizes) ({gap_early:.3f} -> {gap_late:.3f})", gap_late > gap_early + 0.15)
    check(f"final mean fitness high ({res['mean_fitness']:.3f})", res["mean_fitness"] > 0.6)

    # --- most species sit above the threshold in the critical state ---
    frac = bs.fraction_above(res["fitness"], 0.6)
    check(f"most species above threshold ({frac:.2f})", frac > 0.6)

    # --- avalanches span a wide range of sizes (heavy tail, not a narrow band) ---
    # measured relative to a SUBCRITICAL threshold below f_c, where avalanches are bounded and numerous;
    # exactly at f_c the activity almost never ends (one system-spanning avalanche), the mark of criticality.
    sizes = bs.avalanche_sizes(100, 150000, threshold=0.5, seed=1)
    check(f"avalanches occur ({len(sizes)})", len(sizes) > 20)
    check(f"avalanche sizes span a wide range (max {max(sizes)}, min {min(sizes)})",
          max(sizes) > 10 * min(sizes))
    # heavy tail: mean well above the mode/min, many small + few huge
    mean_sz = sum(sizes) / len(sizes)
    check(f"heavy tail (max {max(sizes)} >> mean {mean_sz:.1f})", max(sizes) > 5 * mean_sz)

    # --- critical state independent of initial condition ---
    t1 = bs.simulate(100, 150000, seed=1)["threshold"]
    t2 = bs.simulate(100, 150000, seed=99)["threshold"]
    check(f"threshold seed-independent ({t1:.3f} vs {t2:.3f})", abs(t1 - t2) < 0.05)

    # --- fitness values stay in [0,1) ---
    check("fitness in [0,1)", all(0 <= f < 1 for f in res["fitness"]))

    # --- larger rings give a similar threshold (universality) ---
    t_small = bs.simulate(50, 100000, seed=1)["threshold"]
    t_big = bs.simulate(200, 100000, seed=1)["threshold"]
    check(f"threshold ring-size robust ({t_small:.3f} vs {t_big:.3f})", abs(t_small - t_big) < 0.06)

    # --- critical threshold constant ---
    check("critical threshold ~0.667", abs(bs.critical_threshold_1d() - 0.667) < 0.001)

    # --- deterministic ---
    a = bs.simulate(50, 10000, seed=42)
    b = bs.simulate(50, 10000, seed=42)
    check("deterministic", a["fitness"] == b["fitness"] and a["threshold"] == b["threshold"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
