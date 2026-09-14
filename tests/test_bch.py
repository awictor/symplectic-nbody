"""Tests for binary BCH codes: generator roots, exhaustive t-error correction, dimension, GF(2^m) sanity."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bch import BCH, GF2m, minimal_polynomial  # noqa: E402


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
    # ---- 1. GF(2^m) field axioms --------------------------------------------------------
    gf = GF2m(4)
    check("GF(16) has 15 nonzero elements", gf.n == 15)
    check("GF exp/log inverse", all(gf.log[gf.exp[i]] == i for i in range(gf.n)))
    check("GF multiplicative inverse", all(gf.mul(a, gf.inv(a)) == 1 for a in range(1, 16)))
    check("GF alpha^n == 1 (cyclic)", gf.alpha(gf.n) == 1)

    # ---- 2. BCH(15,7,2): generator is the textbook polynomial ---------------------------
    c = BCH(4, 2)
    check("BCH(15,7,2) length", c.n == 15)
    check("BCH(15,7,2) dimension k=7", c.k == 7, f"k={c.k}")
    check("BCH(15,7,2) generator degree 8", c.deg == 8, f"deg={c.deg}")
    # known generator x^8+x^7+x^6+x^4+1 -> LSB-first [1,0,0,0,1,0,1,1,1]
    check("BCH(15,7,2) generator matches textbook",
          c.g == [1, 0, 0, 0, 1, 0, 1, 1, 1], f"{c.g}")

    # ---- 3. generator vanishes at alpha^1..alpha^{2t} -----------------------------------
    gf = c.gf
    ok = True
    for j in range(1, 2 * c.t + 1):
        aj = gf.alpha(j)
        val = 0
        xp = 1
        for coef in c.g:
            val ^= gf.mul(coef, xp)
            xp = gf.mul(xp, aj)
        if val != 0:
            ok = False
    check("generator roots are alpha^1..alpha^{2t}", ok)

    # ---- 4. codewords are multiples of g => zero syndromes ------------------------------
    msg = [1, 0, 1, 1, 0, 0, 1]
    cw = c.encode(msg)
    check("codeword length n", len(cw) == c.n)
    check("codeword has zero syndromes", all(s == 0 for s in c.syndromes(cw)))
    # systematic: the message sits in the high positions
    check("systematic encoding preserves message", cw[c.deg:c.deg + c.k] == msg)

    # ---- 5. exhaustive: BCH(15,7,2) corrects ALL patterns of <= 2 errors ----------------
    good = 0
    total = 0
    for w in range(3):
        for pos in itertools.combinations(range(c.n), w):
            r = list(cw)
            for p in pos:
                r[p] ^= 1
            dec, ne = c.decode(r)
            total += 1
            if dec == cw and ne == w:
                good += 1
    check("BCH(15,7,2) corrects every <=2-error pattern", good == total, f"{good}/{total}")

    # ---- 6. message recovery after correction -------------------------------------------
    r = list(cw)
    r[2] ^= 1
    r[11] ^= 1
    dm, ne = c.decode_message(r)
    check("decode_message recovers 7 info bits", dm == msg and ne == 2, f"{dm}, {ne}")

    # ---- 7. BCH(15,5,3) corrects all 3-error patterns -----------------------------------
    c3 = BCH(4, 3)
    check("BCH(15,5,3) dimension k=5", c3.k == 5, f"k={c3.k}")
    msg3 = [1, 0, 1, 1, 0]
    cw3 = c3.encode(msg3)
    good3 = 0
    total3 = 0
    for pos in itertools.combinations(range(15), 3):
        r = list(cw3)
        for p in pos:
            r[p] ^= 1
        dec, ne = c3.decode(r)
        total3 += 1
        if dec == cw3:
            good3 += 1
    check("BCH(15,5,3) corrects every 3-error pattern", good3 == total3, f"{good3}/{total3}")

    # ---- 8. zero errors -> zero corrections, codeword unchanged -------------------------
    dec0, ne0 = c.decode(list(cw))
    check("no errors -> zero corrections", dec0 == cw and ne0 == 0)

    # ---- 9. larger code BCH(31,*,3): random t-error trials ------------------------------
    c31 = BCH(5, 3)
    check("BCH(31,...,3) length 31", c31.n == 31)
    rnd = _lcg(2024)
    msg31 = [1 if rnd() > 0.5 else 0 for _ in range(c31.k)]
    cw31 = c31.encode(msg31)
    check("BCH(31) codeword zero syndromes", all(s == 0 for s in c31.syndromes(cw31)))
    trials = 200
    okc = 0
    for _ in range(trials):
        r = list(cw31)
        # inject exactly 3 errors at distinct random positions
        chosen = set()
        while len(chosen) < 3:
            chosen.add(int(rnd() * c31.n) % c31.n)
        for p in chosen:
            r[p] ^= 1
        dec, ne = c31.decode(r)
        if dec == cw31:
            okc += 1
    check("BCH(31,...,3) corrects random 3-error words", okc == trials, f"{okc}/{trials}")

    # ---- 10. minimal polynomial is a factor of the generator ----------------------------
    mp = minimal_polynomial(c.gf, 1)
    # dividing g by mp should give zero remainder
    rem = c._poly_mod(list(c.g), mp)
    check("minimal poly of alpha divides generator", all(x == 0 for x in rem), f"rem {rem}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
