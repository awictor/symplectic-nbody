"""Tests for Hadamard codes: minimum distance n/2, exhaustive error correction, FWHT vs brute decode."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hadamard_code as HC  # noqa: E402


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


def main():
    # ---- 1. every pair of distinct codewords is at distance exactly n/2 -----------------
    for m in (3, 4, 5):
        n = 1 << m
        cws = HC.all_codewords(m)
        dists = set(HC.hamming_distance(cws[a], cws[b])
                    for a, b in itertools.combinations(range(n), 2))
        check(f"m={m}: all pairwise distances == n/2", dists == {n // 2}, f"{sorted(dists)}")
        check(f"m={m}: min_distance == 2^(m-1)", HC.min_distance(m) == n // 2)

    # ---- 2. clean codewords decode to themselves ----------------------------------------
    m = 5
    n = 1 << m
    ok = all(HC.decode(HC.encode(x, m), m) == x for x in range(n))
    check("every clean codeword decodes to itself", ok)

    # ---- 3. exhaustive error correction for small m -------------------------------------
    m = 4
    n = 1 << m
    t = HC.correctable_errors(m)
    msg = 5
    cw = HC.encode(msg, m)
    good = 0
    total = 0
    for w in range(t + 1):
        for positions in itertools.combinations(range(n), w):
            r = list(cw)
            for p in positions:
                r[p] ^= 1
            total += 1
            if HC.decode(r, m) == msg:
                good += 1
    check(f"m=4: corrects every error pattern of weight <= {t}", good == total, f"{good}/{total}")

    # ---- 4. FWHT decoder agrees with brute-force max-correlation ------------------------
    rnd = _lcg(7)
    m = 5
    n = 1 << m
    agree = True
    for _ in range(200):
        x = int(rnd() * n) % n
        r = HC.encode(x, m)
        # inject a random number of errors up to n/4
        ne = int(rnd() * (n // 4))
        pos = set()
        while len(pos) < ne:
            pos.add(int(rnd() * n) % n)
        for p in pos:
            r[p] ^= 1
        if HC.decode(r, m) != HC.brute_decode(r, m):
            agree = False
            break
    check("FWHT decode == brute-force decode", agree)

    # ---- 5. corrects random errors below the radius (larger m) --------------------------
    m = 6
    n = 1 << m
    t = HC.correctable_errors(m)
    rnd = _lcg(99)
    succ = 0
    trials = 200
    for _ in range(trials):
        x = int(rnd() * n) % n
        r = HC.encode(x, m)
        # inject exactly t errors
        pos = set()
        while len(pos) < t:
            pos.add(int(rnd() * n) % n)
        for p in pos:
            r[p] ^= 1
        if HC.decode(r, m) == x:
            succ += 1
    check(f"m=6: corrects exactly t={t} errors every time", succ == trials, f"{succ}/{trials}")

    # ---- 6. encode from bit list matches integer encode ---------------------------------
    check("encode_bits matches encode", HC.encode_bits([1, 0, 1]) == HC.encode(0b101, 3))

    # ---- 7. codeword length and balance -------------------------------------------------
    cw = HC.encode(5, 5)
    check("codeword length 2^m", len(cw) == 32)
    # every nonzero-message codeword is balanced: exactly n/2 ones
    check("nonzero codeword is balanced (n/2 ones)", sum(cw) == 16, f"{sum(cw)}")
    # the all-zero message gives the all-zero codeword
    check("zero message -> all-zero codeword", sum(HC.encode(0, 5)) == 0)

    # ---- 8. confidence peak: clean word gives full peak n -------------------------------
    _, peak = HC.decode_with_confidence(HC.encode(3, 5), 5)
    check("clean codeword FWHT peak == n", peak == 32, f"{peak}")
    # with e errors the peak drops to n - 2e
    r = HC.encode(3, 5)
    r[0] ^= 1
    r[1] ^= 1
    r[2] ^= 1
    _, peak3 = HC.decode_with_confidence(r, 5)
    check("3-error peak == n - 2*3", peak3 == 32 - 6, f"{peak3}")

    # ---- 9. correction radius formula ---------------------------------------------------
    check("correctable_errors(4) == 3", HC.correctable_errors(4) == 3)
    check("correctable_errors(6) == 15", HC.correctable_errors(6) == 15)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
