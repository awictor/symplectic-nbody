"""Tests for KDE: integrates to 1, non-negative, recovers known densities, bandwidth rules."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kde import (  # noqa: E402
    KDE,
    KERNELS,
    silverman_bandwidth,
    scott_bandwidth,
    adaptive_pdf,
    grid,
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


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _normal_samples(n, mu, sigma, rng):
    out = []
    for _ in range(n):
        u1 = max(rng(), 1e-12)
        u2 = rng()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        out.append(mu + sigma * z)
    return out


def main():
    rng = _lcg(2024)

    # ---- 1. integrates to 1 for every kernel --------------------------------------------
    data = _normal_samples(300, 0, 1, rng)
    for kname in KERNELS:
        kde = KDE(data, kernel=kname)
        total = kde.integrate(-8, 8, steps=4000)
        check(f"{kname} KDE integrates to ~1", abs(total - 1) < 0.02, f"{total:.4f}")

    # ---- 2. everywhere non-negative -----------------------------------------------------
    kde = KDE(data)
    check("KDE non-negative on a grid", all(kde.pdf(x) >= 0 for x in grid(-8, 8, 200)))

    # ---- 3. recovers a standard normal --------------------------------------------------
    rng = _lcg(77)
    data = _normal_samples(2000, 0, 1, rng)
    kde = KDE(data)
    # compare to true N(0,1) at a few points
    def true_normal(x):
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    maxerr = max(abs(kde.pdf(x) - true_normal(x)) for x in [-2, -1, 0, 1, 2])
    check("KDE recovers standard normal (max err < 0.04)", maxerr < 0.04, f"{maxerr:.4f}")
    check("peak near x=0", kde.pdf(0) > kde.pdf(1) > kde.pdf(2))

    # ---- 4. bimodal mixture: two peaks ---------------------------------------------------
    rng = _lcg(7)
    data = _normal_samples(1000, -3, 0.6, rng) + _normal_samples(1000, 3, 0.6, rng)
    kde = KDE(data)
    # density at the modes should exceed the valley at 0
    check("bimodal: peaks at +/-3 exceed valley at 0",
          kde.pdf(-3) > kde.pdf(0) and kde.pdf(3) > kde.pdf(0))

    # ---- 5. wider bandwidth is smoother (lower total variation) -------------------------
    rng = _lcg(555)
    data = _normal_samples(200, 0, 1, rng)
    xs = grid(-6, 6, 400)
    def total_variation(kde):
        ys = kde.evaluate(xs)
        return sum(abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1))
    tv_narrow = total_variation(KDE(data, bandwidth=0.1))
    tv_wide = total_variation(KDE(data, bandwidth=1.0))
    check("wider bandwidth -> smoother (less total variation)", tv_wide < tv_narrow,
          f"{tv_narrow:.2f} -> {tv_wide:.2f}")

    # ---- 6. bandwidth rules: positive, shrink with n ------------------------------------
    small = _normal_samples(50, 0, 1, _lcg(1))
    big = _normal_samples(5000, 0, 1, _lcg(1))
    check("Silverman bandwidth positive", silverman_bandwidth(small) > 0)
    check("Scott bandwidth positive", scott_bandwidth(small) > 0)
    check("Silverman shrinks with n", silverman_bandwidth(big) < silverman_bandwidth(small))
    check("Scott shrinks with n", scott_bandwidth(big) < scott_bandwidth(small))
    # for N(0,1) Silverman h ~ 0.9 * n^-0.2, roughly right order
    h = silverman_bandwidth(_normal_samples(1000, 0, 1, _lcg(3)))
    check("Silverman h for N(0,1), n=1000 in reasonable range", 0.1 < h < 0.5, f"{h:.4f}")

    # ---- 7. leave-one-out likelihood peaks near the plug-in bandwidth -------------------
    rng = _lcg(11)
    data = _normal_samples(150, 0, 1, rng)
    h_opt = silverman_bandwidth(data)
    ll_opt = KDE(data, bandwidth=h_opt).loo_log_likelihood()
    ll_tiny = KDE(data, bandwidth=h_opt * 0.15).loo_log_likelihood()
    ll_huge = KDE(data, bandwidth=h_opt * 8).loo_log_likelihood()
    check("LOO likelihood better at plug-in than tiny bandwidth", ll_opt > ll_tiny)
    check("LOO likelihood better at plug-in than huge bandwidth", ll_opt > ll_huge)

    # ---- 8. adaptive KDE integrates to ~1 and is non-negative ---------------------------
    rng = _lcg(321)
    data = _normal_samples(300, 0, 1, rng)
    xs = grid(-8, 8, 500)
    ys = [adaptive_pdf(data, x) for x in xs]
    check("adaptive KDE non-negative", all(y >= 0 for y in ys))
    # trapezoid integral
    integral = sum(0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    check("adaptive KDE integrates to ~1", abs(integral - 1) < 0.05, f"{integral:.4f}")

    # ---- 9. edge cases ------------------------------------------------------------------
    check("single point KDE works", KDE([5.0], bandwidth=1.0).pdf(5.0) > 0)
    try:
        KDE([])
        check("empty data raises", False)
    except ValueError:
        check("empty data raises", True)
    try:
        KDE([1, 2, 3], bandwidth=-1)
        check("negative bandwidth raises", False)
    except ValueError:
        check("negative bandwidth raises", True)
    try:
        KDE([1, 2, 3], kernel="nope")
        check("unknown kernel raises", False)
    except ValueError:
        check("unknown kernel raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
