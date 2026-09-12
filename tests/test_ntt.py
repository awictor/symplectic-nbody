"""Tests for ntt: exact modular convolution vs schoolbook + Python bignum multiplication."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ntt import (ntt, convolve, poly_multiply, multiply_big_integers, schoolbook_convolve, MOD)

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


# --- known convolutions -----------------------------------------------------
check("conv([1,2,3],[4,5,6]) = [4,13,28,27,18]",
      convolve([1, 2, 3], [4, 5, 6]) == [4, 13, 28, 27, 18])
check("poly (1+2x)(1+x) = 1+3x+2x^2", poly_multiply([1, 2], [1, 1]) == [1, 3, 2])
check("empty convolution is empty", convolve([], [1, 2]) == [])
check("single-element convolution", convolve([5], [7]) == [35])

# --- round-trip identity ----------------------------------------------------
rng = LCG(2026)
rt_ok = True
for _ in range(200):
    k = rng.randint(0, 8)
    n = 1 << k
    arr = [rng.randint(0, MOD - 1) for _ in range(n)]
    if ntt(ntt(arr), invert=True) != arr:
        rt_ok = False
        break
check("inverse NTT undoes forward NTT (200 random arrays)", rt_ok)

# --- convolution matches schoolbook (values below MOD) ---------------------
rng = LCG(4242)
conv_ok = True
for _ in range(400):
    la = rng.randint(1, 40)
    lb = rng.randint(1, 40)
    # keep coefficients small so convolution values stay < MOD (exact integer match)
    a = [rng.randint(0, 100) for _ in range(la)]
    b = [rng.randint(0, 100) for _ in range(lb)]
    if convolve(a, b) != schoolbook_convolve(a, b):
        conv_ok = False
        print(f"  mismatch: a={a} b={b}")
        break
check("NTT convolution equals schoolbook (400 random small-coeff pairs)", conv_ok)

# --- convolution is commutative --------------------------------------------
rng = LCG(777)
comm_ok = True
for _ in range(200):
    a = [rng.randint(0, 50) for _ in range(rng.randint(1, 20))]
    b = [rng.randint(0, 50) for _ in range(rng.randint(1, 20))]
    if convolve(a, b) != convolve(b, a):
        comm_ok = False
        break
check("convolution is commutative", comm_ok)

# --- modular agreement even when values exceed MOD -------------------------
rng = LCG(555)
mod_ok = True
for _ in range(200):
    a = [rng.randint(0, MOD - 1) for _ in range(rng.randint(1, 30))]
    b = [rng.randint(0, MOD - 1) for _ in range(rng.randint(1, 30))]
    got = convolve(a, b)
    want = [v % MOD for v in schoolbook_convolve(a, b)]
    if got != want:
        mod_ok = False
        break
check("NTT convolution matches schoolbook reduced mod p (large coefficients)", mod_ok)

# --- big-integer multiplication matches Python's exact product -------------
rng = LCG(31337)
big_ok = True
for _ in range(300):
    x = rng.randint(0, 10 ** 12)
    y = rng.randint(0, 10 ** 12)
    if multiply_big_integers(x, y) != x * y:
        big_ok = False
        print(f"  bignum mismatch: {x} * {y}")
        break
check("NTT big-integer multiply matches Python's exact product (300 pairs)", big_ok)

# --- very large integers ---------------------------------------------------
check("multiply 9^100 * 7^120 exactly", multiply_big_integers(9 ** 100, 7 ** 120) == 9 ** 100 * 7 ** 120)
check("multiply by zero is zero", multiply_big_integers(0, 12345) == 0 and multiply_big_integers(999, 0) == 0)
check("multiply two 500-digit numbers",
      multiply_big_integers(10 ** 500 - 1, 10 ** 500 - 1) == (10 ** 500 - 1) ** 2)

# --- different bases give the same product ---------------------------------
rng = LCG(99)
base_ok = True
for _ in range(100):
    x = rng.randint(0, 10 ** 9)
    y = rng.randint(0, 10 ** 9)
    if multiply_big_integers(x, y, base=100) != x * y:
        base_ok = False
        break
    if multiply_big_integers(x, y, base=1000) != x * y:
        base_ok = False
        break
check("big-integer multiply works in bases 100 and 1000 too", base_ok)

# --- non-power-of-two length is rejected by raw ntt ------------------------
try:
    ntt([1, 2, 3])
    raised = False
except ValueError:
    raised = True
check("raw ntt rejects non-power-of-two length", raised)

# --- agreement with the complex FFT convolution on small integers ----------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fft import convolve as fft_convolve
rng = LCG(1234)
fft_ok = True
for _ in range(100):
    a = [rng.randint(0, 20) for _ in range(rng.randint(1, 12))]
    b = [rng.randint(0, 20) for _ in range(rng.randint(1, 12))]
    ntt_res = convolve(a, b)
    fft_res = [round(v) for v in fft_convolve(a, b)]
    if ntt_res != fft_res:
        fft_ok = False
        break
check("NTT convolution agrees with the complex FFT convolution (small integers)", fft_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all ntt tests passed")
