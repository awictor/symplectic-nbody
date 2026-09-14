"""Validate Fibonacci coding: round-trip, unique '11' terminator, prefix-free, length ~1.44 log2 n, resync."""

import math
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import fibonacci_coding as fc


_failed = 0


def check(name, cond):
    global _failed
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _failed += 1


def main():
    print("Fibonacci coding tests")

    # --- round-trip for a wide range of integers ---
    ok = True
    for n in range(1, 500):
        bits = fc.encode_int(n)
        v, i = fc.decode_int(bits, 0)
        if v != n or i != len(bits):
            ok = False
    check("encode/decode round-trips 1..499", ok)

    # --- every codeword ends in '11' and has no other '11' ---
    term_ok = True
    single_ok = True
    for n in range(1, 300):
        c = fc.encode_int(n)
        if not c.endswith("11"):
            term_ok = False
        # the only "11" is the final terminator: strip last char, no "11" remains
        body = c[:-1]
        if "11" in body:
            single_ok = False
    check("every codeword ends in '11'", term_ok)
    check("'11' appears only as the terminator", single_ok)

    # --- concatenated stream decodes without delimiters ---
    vals = [1, 2, 3, 5, 8, 100, 1, 50, 13, 7]
    bits = fc.encode(vals)
    check("stream round-trips", fc.decode(bits, len(vals)) == vals)
    check("stream decodes to exhaustion", fc.decode(bits) == vals)

    # --- prefix-free: no codeword is a prefix of another ---
    codes = [fc.encode_int(n) for n in range(1, 60)]
    pf = all(not a.startswith(b) for a in codes for b in codes if a != b)
    check("codes are prefix-free", pf)

    # --- code length matches encode_int and grows like ~1.44 log2 n ---
    check("code_length matches encoding", all(fc.code_length(n) == len(fc.encode_int(n)) for n in range(1, 100)))
    # asymptotic slope: length / log2(n) -> ~1.4404 (1/log2(phi))
    n = 10 ** 6
    ratio = fc.code_length(n) / math.log2(n)
    check(f"length ~ 1.44 log2 n (ratio {ratio:.3f})", 1.3 < ratio < 1.6)

    # --- small values: known codewords ---
    # F: 1,2,3,5,8. n=1 -> "1"+term = "11"; n=2 -> "01"+"1"="011"; n=3 -> "001"+"1"
    check("code(1) == '11'", fc.encode_int(1) == "11")
    check("code(2) == '011'", fc.encode_int(2) == "011")
    check("code(3) == '0011'", fc.encode_int(3) == "0011")
    check("code(4) == '1011'", fc.encode_int(4) == "1011")  # 4 = 1 + 3
    check("code(5) == '00011'", fc.encode_int(5) == "00011")

    # --- self-synchronization: a bit flip damages only a bounded neighborhood ---
    vals = list(range(1, 40))
    bits = fc.encode(vals)
    # flip a bit near the front; the TAIL of the stream should still decode correctly
    recovered = fc.resync_after_error(bits, flip_index=5)
    # the last few values should be intact (resynchronized)
    tail_ok = recovered[-10:] == vals[-10:] if len(recovered) >= 10 else False
    check("stream resynchronizes after a bit flip (tail intact)", tail_ok)

    # --- by contrast, most of the stream is fine (few values damaged) ---
    damaged = sum(1 for i in range(min(len(recovered), len(vals))) if recovered[i] != vals[i])
    check(f"bit flip damages only a few codewords ({damaged})", damaged <= 5)

    # --- rejects non-positive ---
    try:
        fc.encode_int(0)
        check("rejects 0", False)
    except ValueError:
        check("rejects 0", True)

    # --- truncated stream raises ---
    try:
        fc.decode_int("0101", 0)   # no "11"
        check("truncated stream raises", False)
    except ValueError:
        check("truncated stream raises", True)

    # --- deterministic ---
    check("deterministic", fc.encode_int(12345) == fc.encode_int(12345))

    print("PASS" if _failed == 0 else f"FAIL ({_failed})")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
