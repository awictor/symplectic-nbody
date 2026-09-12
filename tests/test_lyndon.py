"""Tests for lyndon: Duval factorisation, membership, least rotation, FKM generation vs brute force."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lyndon import (is_lyndon, duval, least_rotation, lyndon_words_up_to, is_lyndon_seq,
                    brute_is_lyndon, brute_least_rotation, brute_lyndon_words)

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


def random_string(rng, n, alphabet):
    return "".join(alphabet[rng.rand() % len(alphabet)] for _ in range(n))


# --- membership -------------------------------------------------------------
check("'aab' is a Lyndon word", is_lyndon("aab"))
check("'a' is a Lyndon word", is_lyndon("a"))
check("'ab' is a Lyndon word", is_lyndon("ab"))
check("'aba' is NOT Lyndon (rotation aab is smaller)", not is_lyndon("aba"))
check("'aa' is NOT Lyndon (ties its rotation)", not is_lyndon("aa"))
check("'ba' is NOT Lyndon", not is_lyndon("ba"))
check("empty string is not Lyndon", not is_lyndon(""))

# --- Duval factorisation properties ----------------------------------------
check("Duval of 'banana' is [b, an, an, a]", duval("banana") == ["b", "an", "an", "a"])
check("Duval of a single Lyndon word is itself", duval("aababc") == ["aababc"])
check("Duval of 'aaa' is [a, a, a]", duval("aaa") == ["a", "a", "a"])

rng = LCG(2026)
factor_ok = True
for _ in range(500):
    s = random_string(rng, rng.randint(1, 16), "abc")
    factors = duval(s)
    # 1) concatenation restores the input
    if "".join(factors) != s:
        factor_ok = False
        print(f"  concat fail: {s!r} -> {factors}")
        break
    # 2) every factor is a Lyndon word
    if not all(is_lyndon(f) for f in factors):
        factor_ok = False
        print(f"  non-Lyndon factor: {s!r} -> {factors}")
        break
    # 3) factors are non-increasing
    if any(factors[i] < factors[i + 1] for i in range(len(factors) - 1)):
        factor_ok = False
        print(f"  not non-increasing: {s!r} -> {factors}")
        break
check("Duval: factors are Lyndon, non-increasing, and concatenate to the input (500 strings)",
      factor_ok)

# --- membership matches brute rotation test --------------------------------
rng = LCG(4242)
mem_ok = True
for _ in range(400):
    s = random_string(rng, rng.randint(1, 12), "ab")
    if is_lyndon(s) != brute_is_lyndon(s):
        mem_ok = False
        break
check("is_lyndon matches the brute rotation definition (400 strings)", mem_ok)

# --- least rotation vs brute -----------------------------------------------
rng = LCG(777)
rot_ok = True
for _ in range(400):
    s = random_string(rng, rng.randint(1, 14), "abc")
    _, got = least_rotation(s)
    _, want = brute_least_rotation(s)
    # the rotation string must match the exhaustive minimum
    if got != want:
        rot_ok = False
        print(f"  rotation mismatch: {s!r} got={got!r} want={want!r}")
        break
check("least rotation matches the exhaustive minimum over all rotations (400 strings)", rot_ok)

# returned index actually produces the returned rotation
rng = LCG(555)
idx_ok = True
for _ in range(200):
    s = random_string(rng, rng.randint(1, 12), "ab")
    idx, rot = least_rotation(s)
    if s[idx:] + s[:idx] != rot:
        idx_ok = False
        break
check("least_rotation index reconstructs the returned rotation", idx_ok)

# a known necklace canonicalisation
check("least rotation of 'cabab' is 'ababc'", least_rotation("cabab")[1] == "ababc")
check("least rotation of 'bca' is 'abc'", least_rotation("bca")[1] == "abc")

# --- FKM generation vs brute -----------------------------------------------
gen_ok = True
for k in range(2, 4):
    for L in range(1, 6):
        got = sorted(lyndon_words_up_to(k, L))
        want = brute_lyndon_words(k, L)
        if got != want:
            gen_ok = False
            print(f"  FKM mismatch k={k} L={L}")
            break
    if not gen_ok:
        break
check("FKM generates exactly the Lyndon words up to length L (brute-checked)", gen_ok)

# generated words are distinct and each is Lyndon
lw = lyndon_words_up_to(3, 5)
check("all generated Lyndon words are genuinely Lyndon", all(is_lyndon_seq(w) for w in lw))
check("generated Lyndon words are distinct", len(lw) == len(set(lw)))

# count of binary Lyndon words of each length matches the necklace-counting formula
# number of Lyndon words of length n over k symbols = (1/n) sum_{d|n} mu(d) k^(n/d)
def mobius(n):
    if n == 1:
        return 1
    result = 1
    d = 2
    m = n
    primes = 0
    while d * d <= m:
        if m % d == 0:
            m //= d
            primes += 1
            if m % d == 0:
                return 0
        else:
            d += 1
    if m > 1:
        primes += 1
    return -1 if primes % 2 else 1


def lyndon_count(n, k):
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += mobius(d) * (k ** (n // d))
    return total // n


count_ok = True
for k in (2, 3):
    lw = lyndon_words_up_to(k, 6)
    from collections import Counter
    by_len = Counter(len(w) for w in lw)
    for n in range(1, 7):
        if by_len[n] != lyndon_count(n, k):
            count_ok = False
            break
check("Lyndon-word counts per length match the Mobius necklace formula", count_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all lyndon tests passed")
