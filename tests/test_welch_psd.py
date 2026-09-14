"""Tests for Welch PSD: tone peak, variance reduction vs periodogram, Parseval, flatness, non-negativity."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import welch_psd as WP  # noqa: E402


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


def _gauss(rnd):
    return math.sqrt(-2 * math.log(rnd() + 1e-12)) * math.cos(2 * math.pi * rnd())


def _cv(vals):
    """Coefficient of variation: std / mean (scale-free smoothness measure)."""
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(var) / m if m > 0 else 0.0


def main():
    fs = 100
    N = 2048

    # ---- 1. a pure tone produces a PSD peak at its frequency ----------------------------
    for f0 in (10, 20, 35):
        sig = [math.cos(2 * math.pi * f0 * n / fs) for n in range(N)]
        freqs, psd = WP.welch(sig, segment_len=128, overlap=0.5, sample_rate=fs)
        peak = freqs[max(range(len(psd)), key=lambda k: psd[k])]
        check(f"tone at {f0} Hz -> PSD peak near {f0}", abs(peak - f0) < 1.0, f"{peak:.2f}")

    # ---- 2. Welch has much lower variance than the raw periodogram (white noise) --------
    rnd = _lcg(42)
    noise = [_gauss(rnd) for _ in range(N)]
    _, pp = WP.periodogram(noise, fs)
    _, pw = WP.welch(noise, segment_len=128, overlap=0.5, sample_rate=fs)
    check("Welch PSD is smoother than the periodogram (lower CV)", _cv(pw) < _cv(pp) * 0.6,
          f"periodogram CV {_cv(pp):.3f} welch CV {_cv(pw):.3f}")

    # ---- 3. more/shorter segments -> more averaging -> lower variance -------------------
    cv_few = _cv(WP.welch(noise, segment_len=256, overlap=0.5, sample_rate=fs)[1])
    cv_many = _cv(WP.welch(noise, segment_len=64, overlap=0.5, sample_rate=fs)[1])
    check("more segments -> lower CV", cv_many < cv_few, f"few {cv_few:.3f} many {cv_many:.3f}")

    # ---- 4. white-noise PSD is approximately flat (no dominant bin) ---------------------
    _, pw = WP.welch(noise, segment_len=128, overlap=0.5, sample_rate=fs)
    mean_psd = sum(pw) / len(pw)
    # no single bin should tower far above the mean for white noise
    check("white-noise PSD is roughly flat", max(pw) < 4 * mean_psd, f"max/mean {max(pw)/mean_psd:.2f}")

    # ---- 5. total power (Parseval): PSD integrates to the tone's power (0.5) -------------
    sig = [math.cos(2 * math.pi * 10 * n / fs) for n in range(N)]
    freqs, psd = WP.welch(sig, segment_len=128, overlap=0.5, sample_rate=fs)
    tp = WP.total_power(freqs, psd, fs)
    check("tone total power ~ 0.5", abs(tp - 0.5) < 0.1, f"{tp:.3f}")

    # ---- 6. white-noise total power ~ variance (unit) -----------------------------------
    freqs, psd = WP.welch(noise, segment_len=128, overlap=0.5, sample_rate=fs)
    tp = WP.total_power(freqs, psd, fs)
    check("white-noise total power ~ 1 (unit variance)", 0.6 < tp < 1.5, f"{tp:.3f}")

    # ---- 7. PSD is non-negative everywhere ----------------------------------------------
    check("PSD is non-negative", all(p >= 0 for p in psd))

    # ---- 8. num_segments counts overlapping segments correctly --------------------------
    check("num_segments with 50% overlap", WP.num_segments(1000, 100, 0.5) == 19,
          f"{WP.num_segments(1000, 100, 0.5)}")
    check("num_segments with no overlap", WP.num_segments(1000, 100, 0.0) == 10)

    # ---- 9. a two-tone signal shows two PSD peaks ---------------------------------------
    two = [math.cos(2 * math.pi * 12 * n / fs) + math.cos(2 * math.pi * 30 * n / fs)
           for n in range(N)]
    freqs, psd = WP.welch(two, segment_len=256, overlap=0.5, sample_rate=fs)
    # find the two tallest bins
    order = sorted(range(len(psd)), key=lambda k: -psd[k])[:2]
    peak_freqs = sorted(freqs[k] for k in order)
    check("two-tone PSD peaks near 12 and 30",
          abs(peak_freqs[0] - 12) < 1.5 and abs(peak_freqs[1] - 30) < 1.5, f"{peak_freqs}")

    # ---- 10. periodogram peak also lands at the tone (single segment sanity) ------------
    freqs, psd = WP.periodogram(sig, fs)
    peak = freqs[max(range(len(psd)), key=lambda k: psd[k])]
    check("periodogram tone peak near 10", abs(peak - 10) < 1.0, f"{peak:.2f}")

    # ---- 11. window choice: Hann and Hamming both locate the tone -----------------------
    for win in ("hann", "hamming", "rectangular"):
        freqs, psd = WP.welch(sig, segment_len=128, overlap=0.5, sample_rate=fs, window=win)
        peak = freqs[max(range(len(psd)), key=lambda k: psd[k])]
        check(f"{win} window locates the 10 Hz tone", abs(peak - 10) < 1.0, f"{peak:.2f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
