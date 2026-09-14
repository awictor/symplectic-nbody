"""Validate multiple-testing corrections: Bonferroni/Holm/BH ordering, FDR control, monotonicity, edge cases."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import benjamini_hochberg as mt


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Multiple-testing correction tests")

    # --- Bonferroni rejects exactly p <= alpha/m ---
    ps = [0.001, 0.008, 0.02, 0.04, 0.5, 0.9]
    m = len(ps)
    alpha = 0.05
    res = mt.bonferroni(ps, alpha)
    expected = [p <= alpha / m for p in ps]
    check("Bonferroni threshold correct", res["reject"] == expected)
    check("Bonferroni adjusted = min(1, p*m)", all(
        abs(res["adjusted"][i] - min(1.0, ps[i] * m)) < 1e-12 for i in range(m)))

    # --- Holm rejects at least as many as Bonferroni ---
    rb = mt.bonferroni(ps, alpha)
    rh = mt.holm(ps, alpha)
    check(f"Holm >= Bonferroni rejections ({rh['n_reject']} >= {rb['n_reject']})",
          rh["n_reject"] >= rb["n_reject"])

    # --- BH rejects at least as many as Holm ---
    rbh = mt.benjamini_hochberg(ps, alpha)
    check(f"BH >= Holm rejections ({rbh['n_reject']} >= {rh['n_reject']})",
          rbh["n_reject"] >= rh["n_reject"])

    # --- known BH example (Benjamini-Hochberg 1995 style) ---
    # p-values sorted; with alpha=0.05, m=10
    ps2 = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.324]
    r = mt.benjamini_hochberg(ps2, 0.05)
    # largest k with p_(k) <= k/10 * 0.05: k=8 (0.0344 <= 0.04); k=9 fails (0.0459 > 0.045).
    # BH rejects the k smallest EVEN THOUGH p_(9) individually fails -- the step-up rule.
    check(f"BH finds correct k ({r['k']})", r["k"] == 8)
    check("BH rejects the 8 smallest", r["reject"][:8] == [True] * 8 and not any(r["reject"][8:]))

    # --- FDR control: mostly-null data with a few strong signals ---
    # 90 nulls (uniform-ish p-values) + 10 strong signals (tiny p)
    class R:
        def __init__(s, seed):
            s.s = seed & 0xFFFFFFFF
        def u(s):
            s.s = (1664525 * s.s + 1013904223) & 0xFFFFFFFF
            return (s.s >> 8) / (1 << 24)
    rng = R(1)
    nulls = [rng.u() for _ in range(90)]        # true nulls: uniform p-values
    signals = [rng.u() * 1e-4 for _ in range(10)]  # true effects: tiny p-values
    allp = nulls + signals
    is_signal = [False] * 90 + [True] * 10
    r = mt.benjamini_hochberg(allp, 0.05)
    # count false discoveries among rejections
    rejected = [i for i in range(100) if r["reject"][i]]
    false_disc = sum(1 for i in rejected if not is_signal[i])
    fdr = false_disc / len(rejected) if rejected else 0.0
    check(f"BH recovers most signals ({sum(1 for i in rejected if is_signal[i])}/10)",
          sum(1 for i in rejected if is_signal[i]) >= 8)
    check(f"BH keeps FDR near alpha (observed {fdr:.3f})", fdr <= 0.10)

    # --- BH more powerful than Bonferroni on this data ---
    rb2 = mt.bonferroni(allp, 0.05)
    check(f"BH more powerful than Bonferroni ({r['n_reject']} >= {rb2['n_reject']})",
          r["n_reject"] >= rb2["n_reject"])

    # --- adjusted p-values monotonic in raw p-values (BH) ---
    order = sorted(range(len(allp)), key=lambda i: allp[i])
    adj_sorted = [r["adjusted"][i] for i in order]
    check("BH q-values monotone non-decreasing", all(adj_sorted[i] <= adj_sorted[i + 1] + 1e-12
                                                      for i in range(len(adj_sorted) - 1)))
    check("BH q-values in [0,1]", all(0 <= q <= 1 for q in r["adjusted"]))

    # --- Holm adjusted monotone and in [0,1] ---
    rh2 = mt.holm(allp, 0.05)
    adj_h = [rh2["adjusted"][i] for i in order]
    check("Holm adjusted monotone", all(adj_h[i] <= adj_h[i + 1] + 1e-12 for i in range(len(adj_h) - 1)))
    check("Holm adjusted in [0,1]", all(0 <= q <= 1 for q in rh2["adjusted"]))

    # --- single test: all methods reduce to the raw p-value decision ---
    for method in ("bonferroni", "holm", "bh"):
        r1 = mt.correct([0.03], method=method, alpha=0.05)
        check(f"single test {method} rejects p<alpha", r1["reject"] == [True])
        r2 = mt.correct([0.08], method=method, alpha=0.05)
        check(f"single test {method} accepts p>alpha", r2["reject"] == [False])

    # --- nothing significant: no rejections, cutoff None ---
    r = mt.benjamini_hochberg([0.4, 0.6, 0.8, 0.99], 0.05)
    check("no signals -> no rejections", r["n_reject"] == 0 and r["cutoff"] is None)

    # --- all significant ---
    r = mt.benjamini_hochberg([0.001, 0.002, 0.003], 0.05)
    check("all tiny -> all rejected", r["n_reject"] == 3)

    # --- deterministic ---
    check("deterministic", mt.benjamini_hochberg(ps, alpha) == mt.benjamini_hochberg(ps, alpha))

    # --- unknown method raises ---
    try:
        mt.correct([0.1], method="nope")
        check("rejects unknown method", False)
    except ValueError:
        check("rejects unknown method", True)

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
