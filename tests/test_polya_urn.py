"""Validate Polya urn: martingale, Beta limiting law, exchangeability, reinforcement spread, symmetry."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import polya_urn as pu


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def mean(xs):
    return sum(xs) / len(xs)


def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def main():
    print("Polya urn tests")

    # --- martingale: expected fraction stays a/(a+b) at every step ---
    a, b, c = 3, 2, 1
    target = pu.expected_fraction(a, b)  # 0.6
    for t in (1, 5, 20, 100):
        fracs = [pu.black_fraction(a, b, c, t, seed=r + 1) for r in range(2000)]
        check(f"E[fraction] ~ a/(a+b) at t={t} ({mean(fracs):.3f} vs {target})",
              abs(mean(fracs) - target) < 0.03)

    # --- limiting fraction matches Beta(a, b) mean and variance (c=1) ---
    a, b = 2, 5
    fracs = [pu.black_fraction(a, b, 1, 2000, seed=r + 1) for r in range(3000)]
    check(f"limit mean ~ Beta mean ({mean(fracs):.3f} vs {pu.beta_mean(a, b):.3f})",
          abs(mean(fracs) - pu.beta_mean(a, b)) < 0.02)
    check(f"limit variance ~ Beta variance ({var(fracs):.4f} vs {pu.beta_variance(a, b):.4f})",
          abs(var(fracs) - pu.beta_variance(a, b)) < 0.01)

    # --- Beta pdf integrates to 1 and matches known moments ---
    # crude numeric integral of the pdf
    N = 2000
    integral = sum(pu.beta_pdf((i + 0.5) / N, 2, 3) / N for i in range(N))
    check("Beta pdf integrates to 1", abs(integral - 1.0) < 0.01)
    # mean via integral
    mean_int = sum(((i + 0.5) / N) * pu.beta_pdf((i + 0.5) / N, 2, 3) / N for i in range(N))
    check("Beta pdf mean matches formula", abs(mean_int - pu.beta_mean(2, 3)) < 0.01)

    # --- exchangeability: all orderings with the same B/W counts are equiprobable ---
    check("exchangeable (2B, 3W)", pu.is_exchangeable(3, 2, 1, 2, 3))
    check("exchangeable (4B, 1W)", pu.is_exchangeable(2, 3, 1, 4, 1))
    check("exchangeable c=2", pu.is_exchangeable(1, 1, 2, 3, 2))

    # --- explicit sequence-probability check: BW and WB have equal probability ---
    p_bw = pu.sequence_probability(["B", "W"], 1, 1, 1)
    p_wb = pu.sequence_probability(["W", "B"], 1, 1, 1)
    check(f"P(BW) == P(WB) ({p_bw:.4f} == {p_wb:.4f})", abs(p_bw - p_wb) < 1e-12)
    # symmetric start a=b=1, c=1: P(BB)=P(WW), and the fraction is uniform on {..}
    check("P(BB) == P(WW) symmetric start",
          abs(pu.sequence_probability(["B", "B"], 1, 1, 1) - pu.sequence_probability(["W", "W"], 1, 1, 1)) < 1e-12)

    # --- symmetric start (a=b): limiting fraction spread symmetrically about 1/2 ---
    # spread seeds widely -- sequential LCG seeds correlate the early (limit-fixing) draws
    fracs = [pu.black_fraction(1, 1, 1, 1000, seed=(r + 1) * 7919) for r in range(4000)]
    check(f"symmetric start mean ~ 1/2 ({mean(fracs):.3f})", abs(mean(fracs) - 0.5) < 0.02)
    # Beta(1,1) is uniform -> variance 1/12
    check(f"symmetric start variance ~ 1/12 ({var(fracs):.3f})", abs(var(fracs) - 1 / 12) < 0.01)

    # --- reinforcement: larger c gives MORE spread at fixed time (stronger rich-get-richer) ---
    fr_c1 = [pu.black_fraction(2, 2, 1, 200, seed=r + 1) for r in range(2000)]
    fr_c5 = [pu.black_fraction(2, 2, 5, 200, seed=r + 1) for r in range(2000)]
    check(f"larger c -> more spread ({var(fr_c1):.4f} < {var(fr_c5):.4f})", var(fr_c5) > var(fr_c1))

    # --- Beta variance formula ---
    check("Beta variance formula", abs(pu.beta_variance(2, 3) - 2 * 3 / (25 * 6)) < 1e-12)

    # --- black_counts monotone non-decreasing (only add balls) ---
    _draws, counts = pu.simulate(3, 2, 1, 100, seed=1)
    check("black counts non-decreasing", all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)))

    # --- deterministic ---
    check("deterministic", pu.simulate(3, 2, 1, 50, seed=42) == pu.simulate(3, 2, 1, 50, seed=42))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
