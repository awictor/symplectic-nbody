"""Tests for Wiener filtering: gain bounds, SNR/MSE improvement, deconvolution, monotone gain."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import wiener_filter as W  # noqa: E402
from bluestein import dft, convolve  # noqa: E402


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


def main():
    # ---- 1. gain H = S/(S+N) lies in [0, 1] ---------------------------------------------
    S = [1.0, 4.0, 0.1, 9.0, 0.01]
    Nz = [1.0, 1.0, 5.0, 0.5, 2.0]
    H = W.wiener_gain(S, Nz)
    check("Wiener gain in [0,1]", all(0.0 <= h <= 1.0 for h in H))

    # ---- 2. no noise -> gain 1; no signal -> gain 0 -------------------------------------
    check("no noise -> gain 1", all(abs(h - 1.0) < 1e-12 for h in W.wiener_gain([1, 2, 3], [0, 0, 0])))
    check("no signal -> gain 0", all(abs(h) < 1e-12 for h in W.wiener_gain([0, 0, 0], [1, 2, 3])))

    # ---- 3. gain is monotone in the local SNR ------------------------------------------
    # for fixed noise, higher signal power -> higher gain
    noise = [1.0] * 5
    sig = [0.1, 0.5, 1.0, 5.0, 20.0]
    g = W.wiener_gain(sig, noise)
    check("gain increases with signal power", all(g[i] < g[i + 1] for i in range(4)))

    # ---- 4. denoising a noisy sinusoid raises the SNR -----------------------------------
    rnd = _lcg(42)
    N = 128
    clean = [math.sin(2 * math.pi * 5 * n / N) + 0.5 * math.sin(2 * math.pi * 12 * n / N)
             for n in range(N)]
    noise = [0.6 * _gauss(rnd) for _ in range(N)]
    noisy = [clean[n] + noise[n] for n in range(N)]
    Spow = [abs(v) ** 2 for v in dft(clean)]
    Npow = [0.6 ** 2 * N] * N
    den = W.denoise(noisy, Spow, Npow)
    check("denoising raises SNR", W.snr(clean, den) > W.snr(clean, noisy) + 3,
          f"noisy {W.snr(clean, noisy):.1f} den {W.snr(clean, den):.1f}")
    check("denoising lowers MSE", W.mse(clean, den) < W.mse(clean, noisy))

    # ---- 5. a clean signal is passed almost unchanged when noise power is ~0 ------------
    Spow = [abs(v) ** 2 for v in dft(clean)]
    den_clean = W.denoise(clean, Spow, [1e-10] * N)
    check("clean signal passes unchanged", W.mse(clean, den_clean) < 1e-6,
          f"{W.mse(clean, den_clean):.2e}")

    # ---- 6. stationary denoiser (unknown signal spectrum) improves SNR ------------------
    rnd = _lcg(7)
    sig = [math.sin(2 * math.pi * 6 * n / N) for n in range(N)]
    noisy = [sig[n] + 0.4 * _gauss(rnd) for n in range(N)]
    den = W.denoise_stationary(noisy, 0.4 ** 2)
    check("stationary denoiser raises SNR", W.snr(sig, den) > W.snr(sig, noisy) + 2,
          f"noisy {W.snr(sig, noisy):.1f} den {W.snr(sig, den):.1f}")

    # ---- 7. Wiener deconvolution beats the blurred signal -------------------------------
    rnd = _lcg(7)
    clean = [math.exp(-((n - 32) ** 2) / 30) for n in range(64)]
    kernel = [math.exp(-((k - 3) ** 2) / 2) for k in range(7)]
    ks = sum(kernel)
    kernel = [k / ks for k in kernel]
    bf = convolve(clean, kernel)
    blurred = [bf[k].real for k in range(64)]
    noisy = [blurred[n] + 0.005 * _gauss(rnd) for n in range(64)]
    rec = W.deconvolve(noisy, kernel, noise_to_signal=0.01)[:64]
    check("deconvolution beats the blurred signal",
          W.mse(clean, rec) < W.mse(clean, blurred),
          f"blurred {W.mse(clean, blurred):.5f} deconv {W.mse(clean, rec):.5f}")

    # ---- 8. deconvolution with a delta kernel is (near) identity ------------------------
    x = [_gauss(rnd) for _ in range(32)]
    rec = W.deconvolve(x, [1.0], noise_to_signal=1e-9)[:32]
    check("deconv with delta kernel ~ identity", W.mse(x, rec) < 1e-6, f"{W.mse(x, rec):.2e}")

    # ---- 9. SNR helper: identical signals -> inf ----------------------------------------
    check("SNR of identical signals is inf", W.snr([1, 2, 3], [1, 2, 3]) == float("inf"))
    check("SNR decreases with more error",
          W.snr([1, 2, 3], [1.1, 2.1, 3.1]) > W.snr([1, 2, 3], [2, 3, 4]))

    # ---- 10. denoise output length matches input ---------------------------------------
    check("denoise preserves length", len(W.denoise(noisy, [1.0] * 64, [1.0] * 64)) == 64)

    # ---- 11. higher noise -> more attenuation (lower total output energy) ---------------
    rnd = _lcg(3)
    sig = [math.sin(2 * math.pi * 4 * n / N) for n in range(N)]
    Spow = [abs(v) ** 2 for v in dft(sig)]
    noisy = [sig[n] + _gauss(rnd) for n in range(N)]
    low_n = W.denoise(noisy, Spow, [0.1 * N] * N)
    high_n = W.denoise(noisy, Spow, [10.0 * N] * N)
    en_low = sum(v * v for v in low_n)
    en_high = sum(v * v for v in high_n)
    check("assuming more noise attenuates more", en_high < en_low,
          f"low-noise energy {en_low:.2f} high-noise {en_high:.2f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
