"""Tests for berlekamp_massey: minimal linear recurrence over Q and GF(2), vs brute force + round-trip."""

import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from berlekamp_massey import (berlekamp_massey, linear_complexity, extend, regenerate,
                              berlekamp_massey_gf2, lfsr_generate, brute_min_recurrence)

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


def as_int(fracs):
    return [int(x) for x in fracs]


# --- known recurrences ------------------------------------------------------
fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
c = berlekamp_massey(fib)
check("Fibonacci recurrence is s[n]=s[n-1]+s[n-2]", c == [Fraction(1), Fraction(1)])
check("Fibonacci linear complexity is 2", linear_complexity(fib) == 2)

trib = [0, 0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
check("Tribonacci recurrence is [1,1,1]",
      berlekamp_massey(trib) == [Fraction(1), Fraction(1), Fraction(1)])

geo = [1, 3, 9, 27, 81, 243, 729]
check("geometric 3^n recurrence is [3]", berlekamp_massey(geo) == [Fraction(3)])

const = [5, 5, 5, 5, 5, 5]
check("constant sequence recurrence is [1]", berlekamp_massey(const) == [Fraction(1)])

check("all-zero sequence has empty recurrence (complexity 0)", berlekamp_massey([0, 0, 0, 0]) == [])
check("empty sequence has empty recurrence", berlekamp_massey([]) == [])

# Pell numbers: s[n] = 2 s[n-1] + s[n-2]
pell = [0, 1, 2, 5, 12, 29, 70, 169, 408]
check("Pell recurrence is [2,1]", berlekamp_massey(pell) == [Fraction(2), Fraction(1)])

# --- round-trip: recurrence regenerates the sequence -----------------------
rng = LCG(2026)
regen_ok = True
for _ in range(300):
    L = rng.randint(1, 5)
    coeffs = [Fraction(rng.randint(-4, 4)) for _ in range(L)]
    if coeffs[-1] == 0:
        coeffs[-1] = Fraction(1)                 # keep the recurrence genuinely order-L
    seed = [Fraction(rng.randint(-5, 5)) for _ in range(L)]
    seq = list(seed)
    for i in range(L, 3 * L + 6):                # enough terms for BM (needs ~2L)
        seq.append(sum(coeffs[j] * seq[i - 1 - j] for j in range(L)))
    found = berlekamp_massey(seq)
    if regenerate(seq, found) != [Fraction(x) for x in seq]:
        regen_ok = False
        break
    if len(found) > L:                           # never longer than the generator
        regen_ok = False
        break
check("recovered recurrence regenerates the sequence and is no longer than the generator", regen_ok)

# --- minimality vs brute force ---------------------------------------------
rng = LCG(4242)
minimal_ok = True
for _ in range(200):
    n = rng.randint(2, 14)
    seq = [Fraction(rng.randint(-3, 3)) for _ in range(n)]
    bm = berlekamp_massey(seq)
    # BM must regenerate the input
    if regenerate(seq, bm) != seq:
        minimal_ok = False
        print(f"  BM does not regenerate: {seq}")
        break
    brute = brute_min_recurrence(seq)
    if brute is not None:
        # brute found a recurrence -> BM's length must be <= brute's (BM is minimal)
        if len(bm) > len(brute):
            minimal_ok = False
            print(f"  BM longer than brute: bm={bm} brute={brute} seq={seq}")
            break
check("BM regenerates the input and is never longer than a brute-found recurrence", minimal_ok)

# --- extrapolation ----------------------------------------------------------
fib = [0, 1, 1, 2, 3, 5, 8, 13]
check("extend continues Fibonacci: 21, 34, 55", as_int(extend(fib, 3)) == [21, 34, 55])

# geometric extrapolation
check("extend continues 2^n", as_int(extend([1, 2, 4, 8, 16], 3)) == [32, 64, 128])

# --- GF(2) LFSR round-trip -------------------------------------------------
rng = LCG(31337)
lfsr_ok = True
for _ in range(200):
    L = rng.randint(1, 6)
    coeffs = [rng.rand() % 2 for _ in range(L)]
    coeffs[-1] = 1                               # tap the last stage so length is exactly L
    seed = [rng.rand() % 2 for _ in range(L)]
    if all(b == 0 for b in seed):
        seed[0] = 1                              # avoid the degenerate all-zero state
    bits = lfsr_generate(coeffs, seed, 4 * L + 8)
    Lrec, crec = berlekamp_massey_gf2(bits)
    # the recovered LFSR must regenerate the same bit-stream
    regen = lfsr_generate(crec, bits[:Lrec], len(bits)) if Lrec > 0 else [0] * len(bits)
    if regen != bits:
        lfsr_ok = False
        print(f"  LFSR mismatch: coeffs={coeffs} seed={seed}")
        break
    if Lrec > L:                                 # complexity never exceeds the generator length
        lfsr_ok = False
        break
check("GF(2): recovered LFSR regenerates the stream, complexity <= generator length", lfsr_ok)

# a maximal-length LFSR: complexity recovered from 2L bits
bits = lfsr_generate([1, 0, 0, 1], [1, 0, 0, 0], 30)   # x^4 + x + 1, period 15
L, coeffs = berlekamp_massey_gf2(bits)
check("GF(2): m-sequence linear complexity recovered as 4", L == 4)
check("GF(2): m-sequence recurrence recovered from a truncated prefix",
      berlekamp_massey_gf2(bits[:8])[0] == 4)

# --- linear complexity of a random stream is about n/2 ---------------------
rng = LCG(99)
bits = [rng.rand() % 2 for _ in range(200)]
L, _ = berlekamp_massey_gf2(bits)
check("random stream has high linear complexity (near n/2)", 80 <= L <= 120)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all berlekamp_massey tests passed")
