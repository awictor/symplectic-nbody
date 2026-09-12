"""Tests for kitamasa: N-th linear-recurrence term vs direct unrolling + known closed forms."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kitamasa import nth_term, nth_term_direct, characteristic_poly

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


# --- known sequences --------------------------------------------------------
check("Fibonacci: F(10)=55", nth_term([1, 1], [0, 1], 10) == 55)
check("Fibonacci: F(20)=6765", nth_term([1, 1], [0, 1], 20) == 6765)
check("Fibonacci: F(50)=12586269025", nth_term([1, 1], [0, 1], 50) == 12586269025)
check("Tribonacci: T(15)=1705", nth_term([1, 1, 1], [0, 0, 1], 15) == 1705)
check("Pell: P(8)=408", nth_term([2, 1], [0, 1], 8) == 408)
check("powers of 2: 2^10=1024", nth_term([2], [1], 10) == 1024)
check("powers of 3: 3^7=2187", nth_term([3], [1], 7) == 2187)
check("constant sequence: s[n]=5", nth_term([1], [5], 100) == 5)

# --- early terms return the initial values ---------------------------------
check("n < k returns initial term", nth_term([1, 1], [3, 7], 0) == 3 and nth_term([1, 1], [3, 7], 1) == 7)

# --- characteristic polynomial ---------------------------------------------
check("Fibonacci char poly is x^2 - x - 1", characteristic_poly([1, 1]) == [1, -1, -1])
check("Pell char poly is x^2 - 2x - 1", characteristic_poly([2, 1]) == [1, -2, -1])

# --- vs direct unrolling on random recurrences -----------------------------
rng = LCG(2026)
direct_ok = True
for _ in range(400):
    k = rng.randint(1, 5)
    coeffs = [rng.randint(-3, 3) for _ in range(k)]
    initial = [rng.randint(-5, 5) for _ in range(k)]
    n = rng.randint(0, 60)
    if nth_term(coeffs, initial, n) != nth_term_direct(coeffs, initial, n):
        direct_ok = False
        print(f"  mismatch: coeffs={coeffs} initial={initial} n={n}")
        break
check("Kitamasa matches direct unrolling (400 random recurrences)", direct_ok)

# --- modular arithmetic matches direct (mod) -------------------------------
rng = LCG(4242)
mod_ok = True
MOD = 10 ** 9 + 7
for _ in range(300):
    k = rng.randint(1, 5)
    coeffs = [rng.randint(0, 20) for _ in range(k)]
    initial = [rng.randint(0, 100) for _ in range(k)]
    n = rng.randint(0, 200)
    if nth_term(coeffs, initial, n, mod=MOD) != nth_term_direct(coeffs, initial, n, mod=MOD):
        mod_ok = False
        break
check("Kitamasa modular results match direct-mod unrolling (300 recurrences)", mod_ok)

# --- huge n: modular Kitamasa matches Python's exact unrolling reduced -----
# Fibonacci at large n, modulo a prime, versus computing the exact F(n) and reducing
def exact_fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


huge_ok = True
for n in (1000, 5000, 12345):
    if nth_term([1, 1], [0, 1], n, mod=MOD) != exact_fib(n) % MOD:
        huge_ok = False
        break
check("modular Fibonacci at large n matches exact-then-reduce", huge_ok)

# --- astronomically large n (only feasible via Kitamasa) -------------------
# just check it runs and is self-consistent under a modulus (F(n+1)^2 - F(n)F(n+2) = (-1)^n)
n = 10 ** 15
fn = nth_term([1, 1], [0, 1], n, mod=MOD)
fn1 = nth_term([1, 1], [0, 1], n + 1, mod=MOD)
fn2 = nth_term([1, 1], [0, 1], n + 2, mod=MOD)
identity = (fn1 * fn1 - fn * fn2) % MOD
expected = (1 if n % 2 == 0 else -1) % MOD
check("Cassini's identity holds for F(10^15) mod p", identity == expected)

# --- Kitamasa is fast for enormous n (would be impossible by unrolling) ----
# a degree-3 recurrence at n = 10^18 -- must return quickly
val = nth_term([1, 1, 1], [0, 0, 1], 10 ** 18, mod=MOD)
check("Tribonacci at n=10^18 mod p computes (a valid residue)", 0 <= val < MOD)

# --- integer (non-modular) exact large term --------------------------------
check("exact Fibonacci F(100) is correct",
      nth_term([1, 1], [0, 1], 100) == 354224848179261915075)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all kitamasa tests passed")
