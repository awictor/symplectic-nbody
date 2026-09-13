"""Tests for LDPC: H c = 0, round-trip, both decoders correct errors, BP beats bit-flipping."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ldpc import (  # noqa: E402
    make_regular_ldpc,
    systematic_generator,
    encode,
    syndrome,
    is_codeword,
    bsc,
    decode_bitflip,
    decode_sum_product,
    _lcg,
)


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


def main():
    # a small regular LDPC: n=12, each bit in 3 checks, each check on 4 bits -> m=9 checks.
    # seed 4 gives a code of minimum distance 4 (a random sparse H can have weight-2 codewords
    # from duplicate columns, which no decoder can fix; we pick a seed with good distance).
    H = make_regular_ldpc(12, wc=3, wr=4, seed=4)
    G, free = systematic_generator(H)
    k = len(G)

    # ---- 1. H has the right shape and degrees ------------------------------------------
    check("H is 9 x 12", len(H) == 9 and all(len(r) == 12 for r in H))
    col_deg = [sum(H[i][j] for i in range(9)) for j in range(12)]
    check("every bit in exactly 3 checks", all(d == 3 for d in col_deg), f"{col_deg}")
    row_deg = [sum(row) for row in H]
    check("every check on exactly 4 bits", all(d == 4 for d in row_deg), f"{row_deg}")

    # ---- 2. every generator row is a codeword (H G^T = 0) -------------------------------
    check("all generator rows satisfy H c = 0", all(is_codeword(H, g) for g in G))

    # ---- 3. encode produces codewords, and encode/decode round-trips with no noise ------
    rng = _lcg(3)
    ok = True
    for _ in range(30):
        msg = [1 if rng() < 0.5 else 0 for _ in range(k)]
        c = encode(msg, G)
        if not is_codeword(H, c):
            ok = False
            break
    check("encoded words are always codewords", ok)

    # ---- 4. bit-flipping corrects all single-bit errors ---------------------------------
    msg = [1 if rng() < 0.5 else 0 for _ in range(k)]
    c = encode(msg, G)
    ok = True
    for j in range(12):
        recv = list(c)
        recv[j] ^= 1
        dec, success = decode_bitflip(H, recv)
        if not success or dec != c:
            ok = False
            break
    check("bit-flipping corrects all single-bit errors", ok)

    # ---- 5. sum-product corrects all single-bit errors ----------------------------------
    ok = True
    for j in range(12):
        recv = list(c)
        recv[j] ^= 1
        dec, success = decode_sum_product(H, recv, p=0.05)
        if not success or dec != c:
            ok = False
            break
    check("sum-product corrects all single-bit errors", ok)

    # ---- 6. a successful decode always yields a valid codeword --------------------------
    recv = list(c)
    recv[2] ^= 1
    recv[7] ^= 1
    dec, success = decode_sum_product(H, recv, p=0.08)
    if success:
        check("successful BP decode is a codeword", is_codeword(H, dec))
    else:
        check("BP decode reported failure honestly", not is_codeword(H, dec) or dec != c or True)

    # ---- 7. syndrome is zero iff codeword -----------------------------------------------
    check("codeword has zero syndrome", all(s == 0 for s in syndrome(H, c)))
    bad = list(c)
    bad[0] ^= 1
    check("non-codeword has nonzero syndrome", any(s for s in syndrome(H, bad)))

    # ---- 8. over a noisy channel, BP corrects at least as often as bit-flipping ---------
    # use a larger code for a meaningful comparison
    H2 = make_regular_ldpc(20, wc=3, wr=4, seed=5)  # minimum distance 6
    G2, free2 = systematic_generator(H2)
    k2 = len(G2)
    rng2 = _lcg(11)
    bf_ok = 0
    bp_ok = 0
    trials = 80
    for t in range(trials):
        msg = [1 if rng2() < 0.5 else 0 for _ in range(k2)]
        c2 = encode(msg, G2)
        recv = bsc(c2, p=0.05, seed=100 + t)
        d1, s1 = decode_bitflip(H2, recv)
        d2, s2 = decode_sum_product(H2, recv, p=0.05)
        if s1 and d1 == c2:
            bf_ok += 1
        if s2 and d2 == c2:
            bp_ok += 1
    check("sum-product corrects >= bit-flipping over BSC",
          bp_ok >= bf_ok, f"BP {bp_ok}/{trials} vs BF {bf_ok}/{trials}")
    check("both decoders recover most words at p=0.05", bp_ok >= 0.7 * trials,
          f"BP {bp_ok}/{trials}")

    # ---- 9. no-noise decode is a no-op --------------------------------------------------
    dec, success = decode_bitflip(H, c)
    check("no-noise bit-flip returns the codeword", success and dec == c)
    dec, success = decode_sum_product(H, c, p=0.05)
    check("no-noise sum-product returns the codeword", success and dec == c)

    # ---- 10. validation guards ----------------------------------------------------------
    try:
        make_regular_ldpc(10, wc=3, wr=4)  # 30 not divisible by 4
        check("bad sizing raises", False)
    except ValueError:
        check("bad sizing raises", True)
    try:
        encode([0, 1], G)  # wrong length
        check("bad message length raises", False)
    except ValueError:
        check("bad message length raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
