"""Tests for the chirp Z-transform: DFT reduction, tone peaks, zoom-FFT sub-bin resolution, linearity."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chirp_z as CZ  # noqa: E402
from bluestein import dft  # noqa: E402


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
    st = seed & 0xFFFFFFFF

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return (st >> 8) / (1 << 24)

    return rnd


def main():
    rnd = _lcg(7)

    # ---- 1. CZT with DFT parameters reduces exactly to the DFT --------------------------
    for n in (8, 13, 16, 31):
        x = [rnd() * 2 - 1 for _ in range(n)]
        c = CZ.czt_as_dft(x)
        d = dft(x)
        err = max(abs(c[k] - d[k]) for k in range(n))
        check(f"CZT == DFT at n={n}", err < 1e-9, f"{err:.2e}")

    # ---- 2. complex input also matches the DFT -----------------------------------------
    x = [complex(rnd(), rnd()) for _ in range(20)]
    err = max(abs(CZ.czt_as_dft(x)[k] - dft(x)[k]) for k in range(20))
    check("CZT == DFT for complex input", err < 1e-9, f"{err:.2e}")

    # ---- 3. a pure tone peaks at the correct DFT bin -----------------------------------
    N = 32
    tone = [math.cos(2 * math.pi * 5 * n / N) for n in range(N)]
    mags = CZ.magnitude(CZ.czt_as_dft(tone))
    peak = max(range(N // 2), key=lambda k: mags[k])
    check("pure tone peaks at bin 5", peak == 5, f"{peak}")

    # ---- 4. zoom-FFT over the full band reproduces the DFT bins -------------------------
    # a zoom from 0 to (N-1)/N * fs with N points should equal the DFT
    N = 24
    x = [rnd() * 2 - 1 for _ in range(N)]
    fs = 1.0
    freqs, spec = CZ.zoom_fft(x, 0.0, (N - 1) / N * fs, N, sample_rate=fs)
    d = dft(x)
    err = max(abs(spec[k] - d[k]) for k in range(N))
    check("zoom over full band == DFT bins", err < 1e-8, f"{err:.2e}")

    # ---- 5. zoom-FFT resolves two tones that fall BETWEEN coarse FFT bins ---------------
    # T = N/fs = 8 s, Rayleigh limit 1/T = 0.125 Hz. Tones 5.1 & 5.4 Hz are 0.3 Hz apart
    # (resolvable) but land between the 0.125-Hz-spaced FFT bins, so a coarse FFT peak-pick
    # would report the wrong frequencies; the fine zoom grid recovers them accurately.
    fs = 32.0
    N = 256
    sig = [math.cos(2 * math.pi * 5.1 * n / fs) + math.cos(2 * math.pi * 5.4 * n / fs)
           for n in range(N)]
    freqs, spec = CZ.zoom_fft(sig, 4.8, 5.8, 300, sample_rate=fs)
    mags = CZ.magnitude(spec)
    mx = max(mags)
    peaks = [freqs[i] for i in range(1, len(mags) - 1)
             if mags[i] > mags[i - 1] and mags[i] > mags[i + 1] and mags[i] > mx * 0.4]
    check("zoom-FFT finds two peaks", len(peaks) == 2, f"{[round(p,3) for p in peaks]}")
    check("zoom peaks are near 5.1 and 5.4",
          any(abs(p - 5.1) < 0.05 for p in peaks) and any(abs(p - 5.4) < 0.05 for p in peaks),
          f"{[round(p,3) for p in peaks]}")

    # ---- 6. sub-bin accuracy: a tone off the FFT grid is located precisely --------------
    fs = 100.0
    N = 200                                        # bin width 0.5 Hz
    f0 = 12.37                                      # deliberately off-grid
    sig = [math.cos(2 * math.pi * f0 * n / fs) for n in range(N)]
    freqs, spec = CZ.zoom_fft(sig, 11.0, 14.0, 600, sample_rate=fs)
    mags = CZ.magnitude(spec)
    peak_f = freqs[max(range(len(mags)), key=lambda i: mags[i])]
    check("zoom locates an off-grid tone to sub-bin accuracy", abs(peak_f - f0) < 0.05,
          f"peak {peak_f:.3f} vs {f0}")

    # ---- 7. constant signal: DC bin holds the sum, others ~0 ----------------------------
    const = [3.0] * 16
    mags = CZ.magnitude(CZ.czt_as_dft(const))
    check("constant signal: DC bin == n*value", abs(mags[0] - 48) < 1e-6, f"{mags[0]}")
    check("constant signal: non-DC bins ~ 0", all(mags[k] < 1e-6 for k in range(1, 16)))

    # ---- 8. linearity: CZT(a x + b y) == a CZT(x) + b CZT(y) -----------------------------
    x = [rnd() for _ in range(12)]
    y = [rnd() for _ in range(12)]
    a, b = 2.0, -3.0
    lhs = CZ.czt_as_dft([a * x[i] + b * y[i] for i in range(12)])
    cx = CZ.czt_as_dft(x)
    cy = CZ.czt_as_dft(y)
    rhs = [a * cx[i] + b * cy[i] for i in range(12)]
    check("CZT is linear", max(abs(lhs[i] - rhs[i]) for i in range(12)) < 1e-9)

    # ---- 9. arbitrary output count M != N -----------------------------------------------
    x = [rnd() for _ in range(10)]
    out = CZ.czt(x, m=25, w=cmath.exp(-2j * cmath.pi / 25), a=1.0)
    check("CZT with M != N returns M points", len(out) == 25)
    # the first 10 of a length-25 unit-circle CZT are NOT the length-10 DFT (different W),
    # but evaluating at the length-10 grid should match: check one point against direct sum
    k = 3
    direct = sum(x[j] * cmath.exp(-2j * cmath.pi * k * j / 25) for j in range(10))
    check("CZT matches direct z-transform sum at a point", abs(out[k] - direct) < 1e-9)

    # ---- 10. magnitude helper -----------------------------------------------------------
    check("magnitude returns |.|", CZ.magnitude([3 + 4j, 0]) == [5.0, 0.0])

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
