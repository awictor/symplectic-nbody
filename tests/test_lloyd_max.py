"""Validate Lloyd-Max: uniform-source optimality, Gaussian SNR gain, optimality conditions, 1/N^2 decay."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import lloyd_max as lm


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Lloyd-Max tests")

    # --- uniform source: optimal levels are evenly spaced ---
    lo, hi = 0.0, 1.0
    uniform_pdf = lambda x: 1.0 if lo <= x <= hi else 0.0
    n = 8
    q = lm.design(uniform_pdf, n, lo, hi)
    # level spacing should be constant
    gaps = [q["levels"][k + 1] - q["levels"][k] for k in range(n - 1)]
    check("uniform source -> evenly-spaced levels",
          max(gaps) - min(gaps) < 1e-3)

    # --- uniform-source distortion matches theoretical Delta^2/12 ---
    delta = (hi - lo) / n
    theory = delta ** 2 / 12.0
    check(f"uniform distortion = Delta^2/12 ({q['distortion']:.2e} vs {theory:.2e})",
          abs(q["distortion"] - theory) < 0.1 * theory)

    # --- Gaussian source: Lloyd-Max beats uniform at same level count ---
    def gauss(x):
        return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    g_lo, g_hi = -5.0, 5.0
    n = 8
    lmq = lm.design(gauss, n, g_lo, g_hi)
    uq = lm.uniform_quantizer(n, g_lo, g_hi)
    snr_lm = lm.snr_db(gauss, lmq, g_lo, g_hi)
    snr_uni = lm.snr_db(gauss, uq, g_lo, g_hi)
    check(f"Lloyd-Max SNR beats uniform ({snr_lm:.2f} dB > {snr_uni:.2f} dB)", snr_lm > snr_uni)
    lm_d = lmq["distortion"]
    uni_d = lm._distortion(gauss, uq["boundaries"], uq["levels"])
    check("Lloyd-Max distortion lower than uniform", lm_d < uni_d)

    # --- Gaussian levels are denser near zero (where mass concentrates) ---
    levels = sorted(lmq["levels"])
    # central gap < outer gap
    central_gap = levels[n // 2] - levels[n // 2 - 1]
    outer_gap = levels[-1] - levels[-2]
    check(f"Gaussian levels denser near zero ({central_gap:.3f} < {outer_gap:.3f})",
          central_gap < outer_gap)

    # --- optimality conditions hold at convergence ---
    # boundaries are midpoints of adjacent levels
    b = lmq["boundaries"]
    lev = lmq["levels"]
    mids_ok = all(abs(b[k] - 0.5 * (lev[k - 1] + lev[k])) < 1e-4 for k in range(1, n))
    check("boundaries are midpoints of adjacent levels", mids_ok)
    # each level is the centroid of its cell
    cents_ok = True
    for k in range(n):
        _m, mean = lm._cell_stats(gauss, b[k], b[k + 1])
        if abs(mean - lev[k]) > 1e-3:
            cents_ok = False
    check("levels are cell centroids", cents_ok)

    # --- distortion decreases with more levels, roughly as 1/N^2 ---
    ds = []
    for nn in (4, 8, 16):
        ds.append(lm.design(gauss, nn, g_lo, g_hi)["distortion"])
    check("more levels -> less distortion", ds[0] > ds[1] > ds[2])
    # doubling N should cut distortion ~4x (1/N^2); allow a broad band
    ratio = ds[0] / ds[1]
    check(f"distortion ~ 1/N^2 (ratio {ratio:.2f} near 4)", 2.5 < ratio < 6.0)

    # --- quantize maps to a valid level ---
    v = lm.quantize(0.3, lmq)
    check("quantized value is one of the levels", any(abs(v - L) < 1e-9 for L in lmq["levels"]))

    # --- empirical (sample-based) Lloyd matches k-means-on-a-line behavior ---
    class R:
        def __init__(s, seed):
            s.s = seed & 0xFFFFFFFF
            s._sp = None

        def normal(s):
            if s._sp is not None:
                v = s._sp
                s._sp = None
                return v
            s.s = (1664525 * s.s + 1013904223) & 0xFFFFFFFF
            u1 = max((s.s >> 8) / (1 << 24), 1e-12)
            s.s = (1664525 * s.s + 1013904223) & 0xFFFFFFFF
            u2 = (s.s >> 8) / (1 << 24)
            r = math.sqrt(-2 * math.log(u1))
            s._sp = r * math.sin(2 * math.pi * u2)
            return r * math.cos(2 * math.pi * u2)
    rng = R(1)
    samples = [rng.normal() for _ in range(2000)]
    emp = lm.design_from_samples(samples, 6)
    # empirical distortion should be positive and levels sorted-ish
    check("empirical Lloyd produces 6 levels", len(emp["levels"]) == 6)
    check("empirical distortion positive", emp["distortion"] > 0)
    # empirical Lloyd beats a uniform partition of the same range on the same samples
    lo_s, hi_s = min(samples), max(samples)
    uq_s = lm.uniform_quantizer(6, lo_s, hi_s)
    uni_emp = sum(min((x - L) ** 2 for L in uq_s["levels"]) for x in samples) / len(samples)
    check("empirical Lloyd beats uniform on samples", emp["distortion"] < uni_emp)

    # --- deterministic ---
    a = lm.design(gauss, 8, g_lo, g_hi)
    c = lm.design(gauss, 8, g_lo, g_hi)
    check("deterministic", a["levels"] == c["levels"])

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
