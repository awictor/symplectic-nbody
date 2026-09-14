"""Validate Burg: AR recovery, guaranteed stability, MEM spectral peaks, resolution, vs Levinson."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import burg_method as burg
import levinson_durbin as lev


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


class _R:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF
        self._spare = None

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)

    def normal(self):
        if self._spare is not None:
            v = self._spare
            self._spare = None
            return v
        u1 = max(self.u(), 1e-12)
        u2 = self.u()
        r = math.sqrt(-2 * math.log(u1))
        self._spare = r * math.sin(2 * math.pi * u2)
        return r * math.cos(2 * math.pi * u2)


def ar_process(coeffs, n, rng, noise=1.0, burn=200):
    """Generate x_t = sum coeffs[i] x_{t-i} + noise. coeffs in x_t = sum a_i x_{t-i} convention."""
    p = len(coeffs)
    x = [0.0] * (n + burn)
    for t in range(p, n + burn):
        x[t] = sum(coeffs[i] * x[t - 1 - i] for i in range(p)) + noise * rng.normal()
    return x[burn:]


def main():
    print("Burg method tests")

    # --- recover AR(2) coefficients from a known stable process ---
    true_coeffs = [0.75, -0.5]   # x_t = 0.75 x_{t-1} - 0.5 x_{t-2} + e
    rng = _R(3)
    x = ar_process(true_coeffs, 400, rng, noise=1.0)
    coeffs, refl, err = burg.burg(x, 2)
    check(f"AR(2) coeffs recovered ({coeffs[0]:.3f},{coeffs[1]:.3f})",
          abs(coeffs[0] - 0.75) < 0.1 and abs(coeffs[1] + 0.5) < 0.1)

    # --- guaranteed stability: all reflection coefficients |k| < 1 ---
    rng = _R(9)
    x = ar_process([0.9, -0.3, 0.2], 300, rng, noise=1.0)
    _c, refl, _e = burg.burg(x, 10)
    check("all reflection coefficients |k| < 1 (stable)", all(abs(k) < 1.0 for k in refl))

    # --- Burg beats Yule-Walker on SHORT records (statistical: average over many seeds) ---
    # The advantage is in expectation, not guaranteed on any single realization -- so average
    # the coefficient error over 40 independent short records.
    true_coeffs = [0.6, -0.7]
    tot_burg = 0.0
    tot_yw = 0.0
    trials = 40
    for seed in range(trials):
        rng = _R(1000 + seed)
        xs = ar_process(true_coeffs, 30, rng, noise=1.0)   # short record
        cb, _, _ = burg.burg(xs, 2)
        cy, _, _ = lev.ar_fit(xs, 2)
        tot_burg += math.hypot(cb[0] - 0.6, cb[1] + 0.7)
        tot_yw += math.hypot(cy[0] - 0.6, cy[1] + 0.7)
    mean_burg = tot_burg / trials
    mean_yw = tot_yw / trials
    check(f"Burg beats Yule-Walker on short records, avg ({mean_burg:.3f} < {mean_yw:.3f})",
          mean_burg < mean_yw)

    # --- MEM spectrum peaks at the true resonant frequency ---
    # signal = sinusoid at 0.15 cycles/sample + light noise
    n = 200
    f0 = 0.15
    rng = _R(5)
    sig = [math.cos(2 * math.pi * f0 * t) + 0.3 * rng.normal() for t in range(n)]
    freqs, psd = burg.me_spectrum(sig, order=20, n_freqs=512)
    peak_idx = max(range(len(psd)), key=lambda i: psd[i])
    peak_f = freqs[peak_idx]
    check(f"MEM PSD peaks at true frequency ({peak_f:.3f} vs {f0})", abs(peak_f - f0) < 0.01)

    # --- resolve two close sinusoids where the periodogram sees one ---
    n = 128
    fa, fb = 0.20, 0.23   # 0.03 apart; FFT bin = 1/128 ~ 0.0078, so ~4 bins -- but with noise + short,
                          # take them closer to make it a real test: 0.20 and 0.215 (~2 bins)
    fa, fb = 0.20, 0.215
    rng = _R(7)
    sig = [math.cos(2 * math.pi * fa * t) + math.cos(2 * math.pi * fb * t) + 0.2 * rng.normal()
           for t in range(n)]
    freqs, psd = burg.me_spectrum(sig, order=30, n_freqs=1024)
    # count local maxima in the band [0.18, 0.235]
    peaks = []
    for i in range(1, len(psd) - 1):
        if 0.18 <= freqs[i] <= 0.235 and psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
            peaks.append(freqs[i])
    check(f"MEM resolves two close tones ({len(peaks)} peaks in band)", len(peaks) >= 2)

    # --- prediction-error variance decreases monotonically with order ---
    rng = _R(11)
    x = ar_process([0.5, -0.4, 0.3], 300, rng, noise=1.0)
    errs = burg.error_vs_order(x, 12)
    check("error variance decreases with order",
          all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1)))

    # --- Burg and Yule-Walker agree on a LONG stationary record ---
    true_coeffs = [0.5, -0.3]
    rng = _R(99)
    x = ar_process(true_coeffs, 5000, rng, noise=1.0)
    cb, _, _ = burg.burg(x, 2)
    cy, _, _ = lev.ar_fit(x, 2)
    check("Burg ~ Yule-Walker on long record",
          abs(cb[0] - cy[0]) < 0.05 and abs(cb[1] - cy[1]) < 0.05)

    # --- one-step prediction is reasonable ---
    p = burg.predict(x, cb)
    check("prediction returns a finite number", math.isfinite(p))

    # --- deterministic ---
    a = burg.burg(x, 4)
    c = burg.burg(x, 4)
    check("deterministic", a[0] == c[0])

    # --- PSD is real and positive ---
    _f, psd = burg.me_spectrum(x, 6, n_freqs=64)
    check("PSD strictly positive", all(p > 0 for p in psd))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
