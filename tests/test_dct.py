"""Tests for dct: fast==direct, orthonormality, Parseval, energy compaction, 2D inversion."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dct import (dct, idct, dct_direct, idct_direct, dct2, idct2,  # noqa: E402
                 energy, compaction_ratio)


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

    # ---- 1. fast DCT matches the direct definition, many lengths ----------------------
    for n in (1, 2, 3, 4, 5, 8, 16, 32, 64, 7, 13):
        x = [rng.u() - 0.5 for _ in range(n)]
        check(f"fast DCT == direct (n={n})", close(dct(x), dct_direct(x), 1e-7),
              f"max err {max((abs(a-b) for a,b in zip(dct(x), dct_direct(x))), default=0):.2e}")

    # ---- 2. inverse recovers the input (idct(dct(x)) == x) ----------------------------
    for n in (1, 2, 4, 8, 16, 32, 5, 11):
        x = [rng.u() * 10 - 5 for _ in range(n)]
        check(f"idct(dct(x)) == x (n={n})", close(idct(dct(x)), x, 1e-7),
              f"max err {max(abs(a-b) for a,b in zip(idct(dct(x)), x)):.2e}")

    # direct inverse too
    x = [rng.u() for _ in range(9)]
    check("idct_direct(dct_direct(x)) == x", close(idct_direct(dct_direct(x)), x, 1e-9))

    # ---- 3. Parseval: energy preserved (orthonormal transform) ------------------------
    for n in (4, 8, 16, 32):
        x = [rng.u() - 0.5 for _ in range(n)]
        check(f"energy preserved sum x^2 == sum X^2 (n={n})",
              abs(energy(x) - energy(dct(x))) < 1e-8,
              f"{energy(x):.6f} vs {energy(dct(x)):.6f}")

    # ---- 4. basis orthonormality: DCT of unit vectors are orthonormal -----------------
    n = 8
    basis = []
    for k in range(n):
        e = [1.0 if i == k else 0.0 for i in range(n)]
        basis.append(dct(e))
    ortho_ok = True
    for i in range(n):
        for j in range(n):
            dp = sum(basis[i][t] * basis[j][t] for t in range(n))
            expected = 1.0 if i == j else 0.0
            if abs(dp - expected) > 1e-8:
                ortho_ok = False
    check("DCT basis is orthonormal", ortho_ok)

    # ---- 5. known 8-point DCT: constant signal -> single DC term ----------------------
    const = [4.0] * 8
    C = dct(const)
    check("constant -> DC only", abs(C[0] - 4.0 * math.sqrt(8)) < 1e-7
          and all(abs(C[k]) < 1e-7 for k in range(1, 8)), f"C0={C[0]:.4f}")

    # a pure cosine at a DCT frequency -> single coefficient
    n = 16
    k0 = 3
    cos_sig = [math.cos(math.pi / n * (i + 0.5) * k0) for i in range(n)]
    Cc = dct(cos_sig)
    peak_ok = all(abs(Cc[k]) < 1e-7 for k in range(n) if k != k0) and abs(Cc[k0]) > 1.0
    check("pure DCT cosine -> single coefficient", peak_ok, f"peak at {k0}: {Cc[k0]:.3f}")

    # ---- 6. energy compaction: smooth signal packs energy into few coefficients -------
    n = 64
    smooth = [math.sin(2 * math.pi * i / n) + 0.5 * math.sin(4 * math.pi * i / n) for i in range(n)]
    r_few = compaction_ratio(smooth, 4)
    check("smooth signal: 4 coeffs capture >98% energy", r_few > 0.98, f"{r_few:.4f}")
    # random signal should NOT compact
    noise = [rng.u() - 0.5 for _ in range(n)]
    r_noise = compaction_ratio(noise, 4)
    check("random signal does not compact into 4 coeffs", r_noise < 0.5, f"{r_noise:.4f}")

    # ---- 7. lossy compression round-trip: keep top coeffs, reconstruct ----------------
    X = dct(smooth)
    # zero all but the 6 largest-magnitude coefficients
    idx = sorted(range(n), key=lambda k: abs(X[k]), reverse=True)[:6]
    Xc = [X[k] if k in set(idx) else 0.0 for k in range(n)]
    recon = idct(Xc)
    rms = math.sqrt(sum((recon[i] - smooth[i]) ** 2 for i in range(n)) / n)
    check("6-coeff reconstruction of smooth signal is accurate", rms < 0.1, f"rms={rms:.4f}")

    # ---- 8. 2D DCT inverts and compacts (the JPEG block transform) --------------------
    block = [[rng.u() * 255 for _ in range(8)] for _ in range(8)]
    recon2 = idct2(dct2(block))
    err2 = max(abs(recon2[i][j] - block[i][j]) for i in range(8) for j in range(8))
    check("2D DCT inverts exactly (8x8)", err2 < 1e-6, f"max err {err2:.2e}")

    # constant block -> only the DC coefficient is nonzero
    cblock = [[7.0] * 8 for _ in range(8)]
    C2 = dct2(cblock)
    dc_only = abs(C2[0][0] - 7.0 * 8) < 1e-6 and all(
        abs(C2[i][j]) < 1e-6 for i in range(8) for j in range(8) if not (i == 0 and j == 0))
    check("constant 8x8 block -> DC coefficient only", dc_only, f"DC={C2[0][0]:.3f}")

    # ---- 9. 2D energy preserved --------------------------------------------------------
    e_in = sum(block[i][j] ** 2 for i in range(8) for j in range(8))
    D = dct2(block)
    e_out = sum(D[i][j] ** 2 for i in range(8) for j in range(8))
    check("2D DCT preserves energy", abs(e_in - e_out) < 1e-6, f"{e_in:.3f} vs {e_out:.3f}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
