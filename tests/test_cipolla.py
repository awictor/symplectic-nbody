"""Tests for Cipolla's modular square root: agreement with Tonelli-Shanks, exactness, non-residues, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cipolla as CP  # noqa: E402
import tonelli_shanks as TS  # noqa: E402


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


def _small_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def main():
    # ---- 1. every returned root squares to n --------------------------------------------
    ok = True
    for n, p in [(10, 13), (2, 7), (5, 41), (3, 11), (17, 101), (44, 97)]:
        r = CP.sqrt_mod(n, p)
        if r is None or not CP.verify(n, r, p):
            ok = False
    check("returned roots square to n (mod p)", ok)

    # ---- 2. non-residues correctly report no root --------------------------------------
    check("sqrt(3) mod 7 is None (non-residue)", CP.sqrt_mod(3, 7) is None)
    check("sqrt(5) mod 7 is None (non-residue)", CP.sqrt_mod(5, 7) is None)

    # ---- 3. agreement with Tonelli-Shanks over a full prime ----------------------------
    for p in (101, 103, 107, 251):
        mism = 0
        for n in range(p):
            c = CP.sqrt_mod(n, p)
            t = TS.sqrt_mod(n, p)
            if (c is None) != (t is None):
                mism += 1
            elif c is not None and (c * c - n) % p != 0:
                mism += 1
        check(f"p={p}: Cipolla consistent with Tonelli-Shanks for all n", mism == 0, f"{mism}")

    # ---- 4. the two roots are negatives of each other ----------------------------------
    roots = CP.both_roots(10, 13)
    check("both_roots returns two roots", roots is not None and len(roots) == 2, f"{roots}")
    check("the two roots are negatives mod p", (roots[0] + roots[1]) % 13 == 0, f"{roots}")
    check("both roots square to n", all(CP.verify(10, r, 13) for r in roots))

    # ---- 5. quadratic-residue detection matches the Legendre symbol ---------------------
    p = 41
    residues = set((x * x) % p for x in range(p))
    ok = all(CP.is_quadratic_residue(n, p) == (n in residues and n != 0) for n in range(1, p))
    check("is_quadratic_residue matches actual residues", ok)

    # ---- 6. edge cases: n = 0 and p = 2 ------------------------------------------------
    check("sqrt(0) mod p == 0", CP.sqrt_mod(0, 13) == 0)
    check("sqrt(1) mod 2 == 1", CP.sqrt_mod(1, 2) == 1)
    check("sqrt(0) mod 2 == 0", CP.sqrt_mod(0, 2) == 0)

    # ---- 7. large prime: exact big-integer square root ---------------------------------
    p = 1000000007
    n = 123456789
    r = CP.sqrt_mod(n, p)
    if r is not None:
        check("large prime: root squares to n", CP.verify(n, r, p), f"r={r}")
    else:
        # n may be a non-residue; verify that claim via the Legendre symbol
        check("large prime: non-residue correctly detected",
              CP.legendre_symbol(n, p) == -1)

    # ---- 8. Cipolla works when p ≡ 1 (mod 8) (large 2-adic valuation, Tonelli's hard case)
    # p = 17 ≡ 1 (mod 16); pick a residue
    p = 17
    residues = [(x * x) % p for x in range(1, p)]
    good = True
    for n in set(residues):
        r = CP.sqrt_mod(n, p)
        if r is None or (r * r - n) % p != 0:
            good = False
    check("handles p with large power of 2 dividing p-1", good)

    # ---- 9. every quadratic residue has a root, every non-residue has none -------------
    p = 97
    all_ok = True
    for n in range(1, p):
        is_qr = CP.is_quadratic_residue(n, p)
        r = CP.sqrt_mod(n, p)
        if is_qr and r is None:
            all_ok = False
        if not is_qr and r is not None:
            all_ok = False
    check("residues <-> roots exactly (p=97)", all_ok)

    # ---- 10. randomized: square a random x, recover a valid root ------------------------
    st = 12345

    def rnd():
        nonlocal st
        st = (1664525 * st + 1013904223) & 0xFFFFFFFF
        return st

    p = 7919
    ok = True
    for _ in range(200):
        x = rnd() % p
        n = (x * x) % p
        r = CP.sqrt_mod(n, p)
        if r is None or (r * r - n) % p != 0:
            ok = False
            break
    check("randomized: squares always have recoverable roots", ok)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
