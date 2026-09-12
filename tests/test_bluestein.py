"""Tests for bluestein: arbitrary-length DFT vs direct definition, round trip, prime lengths."""

import cmath
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bluestein import (dft, idft, dft_direct, convolve, convolve_direct,  # noqa: E402
                       _fft_pow2, _next_pow2)


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


def close_seq(a, b, tol=1e-8):
    if len(a) != len(b):
        return False
    return all(abs(x - y) < tol for x, y in zip(a, b))


def main():
    rng = LCG(2024)

    # ---- 1. Bluestein DFT matches the direct definition at MANY lengths ---------------
    lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 15, 16, 17, 31, 32, 64, 100, 101]
    for n in lengths:
        x = [complex(rng.u() - 0.5, rng.u() - 0.5) for _ in range(n)]
        fast = dft(x)
        ref = dft_direct(x)
        check(f"DFT matches direct definition (n={n})", close_seq(fast, ref, 1e-7),
              f"max err {max((abs(a-b) for a,b in zip(fast,ref)), default=0):.2e}")

    # ---- 2. prime lengths specifically (radix-2 can't do these) -----------------------
    for p in (7, 13, 101, 251):
        x = [complex(rng.u() - 0.5, rng.u() - 0.5) for _ in range(p)]
        check(f"prime-length DFT correct (n={p})", close_seq(dft(x), dft_direct(x), 1e-6))

    # ---- 3. round trip idft(dft(x)) == x ----------------------------------------------
    for n in (5, 7, 12, 13, 100, 101):
        x = [complex(rng.u() - 0.5, rng.u() - 0.5) for _ in range(n)]
        back = idft(dft(x))
        check(f"round trip idft(dft(x)) == x (n={n})", close_seq(back, x, 1e-7),
              f"max err {max(abs(a-b) for a,b in zip(back,x)):.2e}")

    # real input round trip returns (nearly) real
    xr = [rng.u() for _ in range(11)]
    backr = idft(dft(xr))
    check("real input recovered", all(abs(backr[i].real - xr[i]) < 1e-8 and abs(backr[i].imag) < 1e-8
                                      for i in range(11)))

    # ---- 4. known transforms ----------------------------------------------------------
    # constant sequence -> spike at bin 0 of value n, zero elsewhere
    n = 13
    const = [3.0] * n
    C = dft(const)
    check("constant -> spike at bin 0", abs(C[0] - 3.0 * n) < 1e-7
          and all(abs(C[k]) < 1e-7 for k in range(1, n)))

    # pure exponential of integer frequency f -> single nonzero bin at f
    f = 3
    n = 17
    expo = [cmath.exp(2j * math.pi * f * j / n) for j in range(n)]
    E = dft(expo)
    peak_ok = abs(E[f] - n) < 1e-6 and all(abs(E[k]) < 1e-6 for k in range(n) if k != f)
    check("integer-frequency exponential -> single bin", peak_ok,
          f"E[{f}]={E[f]:.3f}")

    # ---- 5. linearity ------------------------------------------------------------------
    a = [complex(rng.u(), rng.u()) for _ in range(9)]
    b = [complex(rng.u(), rng.u()) for _ in range(9)]
    lhs = dft([2 * a[i] + 3 * b[i] for i in range(9)])
    da, db = dft(a), dft(b)
    rhs = [2 * da[i] + 3 * db[i] for i in range(9)]
    check("DFT is linear", close_seq(lhs, rhs, 1e-8))

    # ---- 6. convolution matches naive --------------------------------------------------
    for (la, lb) in [(5, 3), (7, 7), (10, 4), (13, 6)]:
        aa = [rng.u() - 0.5 for _ in range(la)]
        bb = [rng.u() - 0.5 for _ in range(lb)]
        fast = convolve(aa, bb)
        ref = convolve_direct(aa, bb)
        check(f"convolution matches naive ({la}x{lb})", close_seq(fast, ref, 1e-7),
              f"max err {max(abs(x-y) for x,y in zip(fast,ref)):.2e}")

    # ---- 7. helper sanity --------------------------------------------------------------
    check("_next_pow2 correct", _next_pow2(1) == 1 and _next_pow2(5) == 8 and _next_pow2(16) == 16
          and _next_pow2(17) == 32)
    # radix-2 FFT self-check vs direct on a power of two
    xp = [complex(rng.u(), rng.u()) for _ in range(8)]
    check("internal radix-2 FFT matches direct", close_seq(_fft_pow2(xp), dft_direct(xp), 1e-9))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
