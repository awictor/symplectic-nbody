"""Tests for wavelet_transform: perfect reconstruction, Parseval, vanishing moments, denoising."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wavelet_transform import (dwt_step, idwt_step, dwt, idwt, dwt2_step, idwt2_step,  # noqa: E402
                               denoise, threshold_details, coeffs_to_flat, energy)


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def u(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return (self.s >> 8) / (1 << 24)


def close(a, b, tol=1e-8):
    return len(a) == len(b) and all(abs(x - y) < tol for x, y in zip(a, b))


def main():
    rng = LCG(2024)

    # ---- 1. single-level perfect reconstruction, both wavelets ------------------------
    for wav in ("haar", "db4"):
        for n in (2, 4, 8, 16, 64):
            x = [rng.u() * 10 - 5 for _ in range(n)]
            a, d = dwt_step(x, wav)
            check(f"single-level reconstruction ({wav}, n={n})",
                  close(idwt_step(a, d, wav), x, 1e-8),
                  f"max err {max(abs(p-q) for p,q in zip(idwt_step(a,d,wav), x)):.2e}")

    # ---- 2. multi-level perfect reconstruction ----------------------------------------
    for wav in ("haar", "db4"):
        for n in (8, 16, 32, 64, 128):
            x = [rng.u() * 10 for _ in range(n)]
            approx, details = dwt(x, wav)
            check(f"multi-level reconstruction ({wav}, n={n})", close(idwt(approx, details, wav), x, 1e-7))

    # ---- 3. Parseval energy preservation ----------------------------------------------
    for wav in ("haar", "db4"):
        x = [rng.u() - 0.5 for _ in range(64)]
        approx, details = dwt(x, wav)
        flat = coeffs_to_flat(approx, details)
        check(f"energy preserved ({wav})", abs(energy(x) - energy(flat)) < 1e-8,
              f"{energy(x):.6f} vs {energy(flat):.6f}")

    # ---- 4. flat packing has the same length --------------------------------------------
    x = [rng.u() for _ in range(32)]
    approx, details = dwt(x, "haar")
    check("flat coefficients same length as input", len(coeffs_to_flat(approx, details)) == 32)

    # ---- 5. vanishing moments --------------------------------------------------------
    # Haar kills a constant's detail; db4 kills a constant AND a linear ramp
    const = [3.0] * 64
    _, dH = dwt_step(const, "haar")
    check("Haar: constant -> zero detail", all(abs(c) < 1e-9 for c in dH), f"max {max(abs(c) for c in dH):.2e}")

    ramp = [0.5 * i for i in range(64)]
    _, dR_haar = dwt_step(ramp, "haar")
    _, dR_db4 = dwt_step(ramp, "db4")
    # db4 detail on a ramp should be ~0 (interior); Haar detail on a ramp is a nonzero constant
    interior_db4 = max(abs(dR_db4[i]) for i in range(1, len(dR_db4) - 1))
    check("db4: linear ramp -> ~zero detail (2 vanishing moments)", interior_db4 < 1e-9,
          f"interior max {interior_db4:.2e}")
    check("Haar: linear ramp -> nonzero detail (only 1 vanishing moment)",
          max(abs(c) for c in dR_haar) > 0.1)

    # ---- 6. sparsity / compaction on a smooth signal ----------------------------------
    n = 128
    smooth = [math.sin(2 * math.pi * i / n) + 0.3 * math.sin(6 * math.pi * i / n) for i in range(n)]
    approx, details = dwt(smooth, "db4")
    flat = coeffs_to_flat(approx, details)
    mags = sorted((abs(c) for c in flat), reverse=True)
    top16_energy = sum(m * m for m in mags[:16]) / energy(flat)
    check("smooth signal: top 16 coeffs hold >95% energy", top16_energy > 0.95, f"{top16_energy:.4f}")

    # ---- 7. denoising lowers error toward the clean signal ----------------------------
    clean = [math.sin(2 * math.pi * i / n) for i in range(n)]
    noisy = [clean[i] + 0.3 * (rng.u() - 0.5) for i in range(n)]
    den = denoise(noisy, "db4", threshold=0.15)
    err_noisy = math.sqrt(sum((noisy[i] - clean[i]) ** 2 for i in range(n)) / n)
    err_den = math.sqrt(sum((den[i] - clean[i]) ** 2 for i in range(n)) / n)
    check("wavelet denoising reduces error", err_den < err_noisy, f"noisy={err_noisy:.4f} den={err_den:.4f}")

    # ---- 8. 2D transform inverts exactly ----------------------------------------------
    img = [[rng.u() * 255 for _ in range(8)] for _ in range(8)]
    LL, LH, HL, HH = dwt2_step(img, "haar")
    recon = idwt2_step(LL, LH, HL, HH, "haar")
    err2 = max(abs(recon[i][j] - img[i][j]) for i in range(8) for j in range(8))
    check("2D DWT inverts exactly (Haar)", err2 < 1e-7, f"max err {err2:.2e}")

    LL, LH, HL, HH = dwt2_step(img, "db4")
    recon = idwt2_step(LL, LH, HL, HH, "db4")
    err2 = max(abs(recon[i][j] - img[i][j]) for i in range(8) for j in range(8))
    check("2D DWT inverts exactly (db4)", err2 < 1e-7, f"max err {err2:.2e}")

    # constant image -> only LL nonzero
    cimg = [[5.0] * 8 for _ in range(8)]
    LL, LH, HL, HH = dwt2_step(cimg, "haar")
    detail_max = max(max(max(abs(v) for v in row) for row in band) for band in (LH, HL, HH))
    check("constant image -> zero detail bands", detail_max < 1e-9, f"{detail_max:.2e}")

    # ---- 9. hand-checked Haar values --------------------------------------------------
    # Haar of [1,3]: approx = (1+3)/sqrt(2), detail = (1-3)/sqrt(2)... with our g=(-1)^k h_{1-k}
    a, d = dwt_step([1.0, 3.0], "haar")
    check("Haar of [1,3] approx = 4/sqrt(2)", abs(a[0] - 4 / math.sqrt(2)) < 1e-9, f"{a[0]}")
    check("Haar of [1,3] |detail| = 2/sqrt(2)", abs(abs(d[0]) - 2 / math.sqrt(2)) < 1e-9, f"{d[0]}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
