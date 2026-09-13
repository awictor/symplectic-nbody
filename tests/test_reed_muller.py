"""Tests for RM(1,m): round-trip, error correction to radius, min distance, FWHT == brute decode."""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reed_muller import (  # noqa: E402
    encode,
    decode,
    code_parameters,
    correctable_errors,
    hamming_distance,
    all_codewords,
    brute_decode,
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
    # ---- 1. encode-then-decode round-trips every message --------------------------------
    ok = True
    for m in [2, 3, 4, 5]:
        for msg_int in range(1 << (m + 1)):
            message = [(msg_int >> i) & 1 for i in range(m + 1)]
            word = encode(message, m)
            decoded, _ = decode(word, m)
            if decoded != message:
                ok = False
    check("encode->decode round-trips all messages (m=2..5)", ok)

    # ---- 2. code parameters -------------------------------------------------------------
    check("RM(1,5) is [32, 6, 16]", code_parameters(5) == (32, 6, 16, 7))
    check("RM(1,4) is [16, 5, 8]", code_parameters(4) == (16, 5, 8, 3))
    check("codeword length = 2^m", all(code_parameters(m)[0] == (1 << m) for m in range(2, 7)))

    # ---- 3. corrects any error pattern up to the radius (exhaustive, small m) -----------
    for m in [3, 4]:
        n, k, d, t = code_parameters(m)
        message = [1, 0, 1] + [0] * (m - 2) if m >= 2 else [1]
        message = message[:m + 1] + [0] * (m + 1 - len(message))
        message = message[:m + 1]
        word = encode(message, m)
        ok = True
        # all error patterns of weight up to t
        for w in range(t + 1):
            for positions in itertools.combinations(range(n), w):
                rec = list(word)
                for pos in positions:
                    rec[pos] ^= 1
                decoded, _ = decode(rec, m)
                if decoded != message:
                    ok = False
                    break
            if not ok:
                break
        check(f"RM(1,{m}) corrects all errors up to t={t}", ok)

    # ---- 4. minimum distance is 2^(m-1) -------------------------------------------------
    for m in [3, 4]:
        n, k, d, t = code_parameters(m)
        words = [w for _, w in all_codewords(m)]
        mind = None
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                dist = hamming_distance(words[i], words[j])
                if mind is None or dist < mind:
                    mind = dist
        check(f"RM(1,{m}) minimum distance = {d}", mind == d, f"{mind}")

    # ---- 5. FWHT decoder agrees with brute-force nearest codeword -----------------------
    def _lcg(seed):
        st = seed & 0xFFFFFFFF

        def nxt():
            nonlocal st
            st = (1664525 * st + 1013904223) & 0xFFFFFFFF
            return (st >> 8) / (1 << 24)
        return nxt
    rng = _lcg(2024)
    mism = 0
    for m in [3, 4]:
        n, k, d, t = code_parameters(m)
        for _ in range(100):
            msg_int = int(rng() * (1 << (m + 1)))
            message = [(msg_int >> i) & 1 for i in range(m + 1)]
            word = encode(message, m)
            rec = list(word)
            # add up to t errors
            for _ in range(int(rng() * (t + 1))):
                rec[int(rng() * n)] ^= 1
            fwht_dec, _ = decode(rec, m)
            brute_dec = brute_decode(rec, m)
            if fwht_dec != brute_dec:
                mism += 1
    check("FWHT decoder == brute nearest-codeword", mism == 0, f"{mism} mismatches")

    # ---- 6. correction radius helper ----------------------------------------------------
    check("correctable_errors(5) = 7", correctable_errors(5) == 7)
    check("correctable_errors(4) = 3", correctable_errors(4) == 3)

    # ---- 7. edge cases ------------------------------------------------------------------
    # all-zero message -> all-zero codeword
    check("zero message -> zero codeword", encode([0, 0, 0], 2) == [0] * 4)
    # a_0 = 1 alone -> all-ones codeword
    check("constant message -> all-ones", encode([1, 0, 0], 2) == [1] * 4)
    try:
        encode([1, 0], 3)  # wrong message length
        check("bad message length raises", False)
    except ValueError:
        check("bad message length raises", True)
    try:
        decode([0, 1, 0], 3)  # wrong word length
        check("bad word length raises", False)
    except ValueError:
        check("bad word length raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
