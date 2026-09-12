"""Tests for karatsuba: Karatsuba/Toom-3 multiplication vs Python bignum + schoolbook."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from karatsuba import (karatsuba, toom3, poly_multiply_karatsuba, schoolbook_multiply,
                       _poly_schoolbook)

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

    def randbig(self, max_digits):
        digits = self.randint(1, max_digits)
        val = 0
        for _ in range(digits):
            val = val * 10 + self.randint(0, 9)
        return val


# --- known / edge cases -----------------------------------------------------
check("karatsuba 6*7 = 42", karatsuba(6, 7) == 42)
check("karatsuba 0 * anything = 0", karatsuba(0, 123456789) == 0)
check("karatsuba 1 * n = n", karatsuba(1, 987654321) == 987654321)
check("toom3 6*7 = 42", toom3(6, 7) == 42)
check("karatsuba negatives", karatsuba(-12345, 678) == -12345 * 678)
check("karatsuba both negative", karatsuba(-999, -111) == 999 * 111)
check("toom3 mixed sign", toom3(-10 ** 40, 10 ** 40) == -(10 ** 80))

# --- Karatsuba vs Python bignum --------------------------------------------
rng = LCG(2026)
kara_ok = True
for _ in range(400):
    x = rng.randbig(rng.randint(1, 120))
    y = rng.randbig(rng.randint(1, 120))
    sx = -1 if rng.rand() % 2 else 1
    sy = -1 if rng.rand() % 2 else 1
    x *= sx
    y *= sy
    if karatsuba(x, y) != x * y:
        kara_ok = False
        print(f"  karatsuba mismatch: {x} * {y}")
        break
check("Karatsuba matches Python's exact product (400 random, incl negatives)", kara_ok)

# --- Toom-3 vs Python bignum -----------------------------------------------
rng = LCG(4242)
toom_ok = True
for _ in range(400):
    x = rng.randbig(rng.randint(1, 150))
    y = rng.randbig(rng.randint(1, 150))
    if toom3(x, y) != x * y:
        toom_ok = False
        print(f"  toom3 mismatch: {x} * {y}")
        break
check("Toom-3 matches Python's exact product (400 random)", toom_ok)

# --- both agree with the schoolbook limb reference -------------------------
rng = LCG(777)
school_ok = True
for _ in range(200):
    x = rng.randbig(rng.randint(1, 100))
    y = rng.randbig(rng.randint(1, 100))
    ref = schoolbook_multiply(x, y)
    if karatsuba(x, y) != ref or toom3(x, y) != ref:
        school_ok = False
        break
check("Karatsuba and Toom-3 agree with the schoolbook limb reference (200 inputs)", school_ok)

# --- very large numbers ----------------------------------------------------
check("karatsuba on 1000-digit numbers",
      karatsuba(10 ** 1000 - 1, 10 ** 1000 - 7) == (10 ** 1000 - 1) * (10 ** 1000 - 7))
check("toom3 on 1000-digit numbers",
      toom3(10 ** 1000 - 1, 10 ** 1000 - 7) == (10 ** 1000 - 1) * (10 ** 1000 - 7))
check("karatsuba 9^500 * 7^600", karatsuba(9 ** 500, 7 ** 600) == 9 ** 500 * 7 ** 600)
check("karatsuba equals toom3 on a big shared input",
      karatsuba(3 ** 400, 5 ** 400) == toom3(3 ** 400, 5 ** 400) == 3 ** 400 * 5 ** 400)

# --- polynomial multiplication vs direct convolution -----------------------
rng = LCG(31337)
poly_ok = True
for _ in range(300):
    a = [rng.randint(-20, 20) for _ in range(rng.randint(1, 60))]
    b = [rng.randint(-20, 20) for _ in range(rng.randint(1, 60))]
    if poly_multiply_karatsuba(a, b) != _poly_schoolbook(a, b):
        poly_ok = False
        print(f"  poly mismatch: a={a} b={b}")
        break
check("Karatsuba polynomial multiply equals direct convolution (300 pairs)", poly_ok)

# --- commutativity ----------------------------------------------------------
rng = LCG(555)
comm_ok = True
for _ in range(200):
    x = rng.randbig(rng.randint(1, 80))
    y = rng.randbig(rng.randint(1, 80))
    if karatsuba(x, y) != karatsuba(y, x) or toom3(x, y) != toom3(y, x):
        comm_ok = False
        break
check("multiplication is commutative for both methods", comm_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all karatsuba tests passed")
