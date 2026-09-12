"""Tests for golay: corrects all <=3-error patterns, min distance 8, weight enumerator, linearity."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from golay import (encode, decode, corrupt, weight_enumerator, minimum_distance,  # noqa: E402
                   _syndrome, _popcount, _LCG)


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
    # ---- 1. encode/decode round trip on a clean channel -------------------------------
    rt = all(decode(encode(m))[0] == m for m in range(4096))
    check("clean-channel round trip for all 4096 messages", rt)

    # ---- 2. minimum distance is 8 -----------------------------------------------------
    check("minimum distance is 8", minimum_distance() == 8, f"{minimum_distance()}")

    # ---- 3. weight enumerator matches the known Golay distribution --------------------
    we = weight_enumerator()
    expected = {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}
    check("weight enumerator matches Golay (1,759,2576,759,1)", we == expected, f"{we}")

    # ---- 4. corrects EVERY error pattern of weight 0,1,2,3 on several messages --------
    # exhaustive over all C(24,k) positions for a sample of messages (full sweep is huge)
    messages = [0, 1, 42, 0xABC, 0xFFF, 0x555, 0xAAA, 2730]
    fail3 = 0
    for m in messages:
        cw = encode(m)
        # weight 0
        if decode(cw)[0] != m:
            fail3 += 1
        # weight 1
        for i in range(24):
            if decode(cw ^ (1 << i))[0] != m:
                fail3 += 1
        # weight 2
        for i in range(24):
            for j in range(i + 1, 24):
                if decode(cw ^ (1 << i) ^ (1 << j))[0] != m:
                    fail3 += 1
        # weight 3
        for i in range(24):
            for j in range(i + 1, 24):
                for k in range(j + 1, 24):
                    if decode(cw ^ (1 << i) ^ (1 << j) ^ (1 << k))[0] != m:
                        fail3 += 1
    check("corrects ALL weight-0..3 errors (exhaustive on 8 messages)", fail3 == 0,
          f"{fail3} failures")

    # ---- 5. errors_corrected count is reported accurately -----------------------------
    m = 0xABC
    cw = encode(m)
    counts_ok = True
    for ne, e in ((0, 0), (1, 0b1), (2, 0b101), (3, 0b10101)):
        msg, nc = decode(cw ^ e)
        if msg != m or nc != _popcount(e):
            counts_ok = False
    check("errors_corrected count is accurate", counts_ok)

    # ---- 6. a weight-4 error is detected (not silently miscorrected as the true msg) --
    # for a perfect-ish code, some weight-4 patterns are uncorrectable; check detection happens
    m = 0x555
    cw = encode(m)
    detected = 0
    miscorrect_to_wrong = 0
    n4 = 0
    rng = _LCG(7)
    for _ in range(500):
        bad = corrupt(cw, 4, rng)
        msg, nc = decode(bad)
        n4 += 1
        if msg is None:
            detected += 1
        elif msg != m:
            miscorrect_to_wrong += 1
    # weight-4 errors are never silently decoded back to the ORIGINAL message (distance 8 => impossible)
    check("weight-4 error never decodes to the original message",
          all(decode(corrupt(encode(0x555), 4, _LCG(s)))[0] != 0x555 or
              decode(corrupt(encode(0x555), 4, _LCG(s)))[0] is None for s in range(50)))
    check("weight-4 errors are flagged as uncorrectable at least sometimes", detected > 0,
          f"{detected}/{n4} detected")

    # ---- 7. random-channel correction over many trials --------------------------------
    rng = _LCG(2024)
    bad = 0
    trials = 5000
    for _ in range(trials):
        m = rng.randint(4096)
        ne = rng.randint(4)             # 0..3 errors
        recv = corrupt(encode(m), ne, rng)
        if decode(recv)[0] != m:
            bad += 1
    check("random channel: all <=3-error transmissions recovered", bad == 0, f"{bad}/{trials}")

    # ---- 8. linearity: sum of two codewords is a codeword -----------------------------
    lin_ok = True
    rng = _LCG(11)
    for _ in range(200):
        a = rng.randint(4096)
        b = rng.randint(4096)
        ca = encode(a)
        cb = encode(b)
        # XOR of codewords should be the codeword of (a XOR b)
        if (ca ^ cb) != encode(a ^ b):
            lin_ok = False
    check("code is linear (codeword XOR is a codeword)", lin_ok)

    # ---- 9. syndrome of a valid codeword is zero --------------------------------------
    check("valid codewords have zero syndrome", all(_syndrome(encode(m)) == 0 for m in range(0, 4096, 17)))

    # ---- 10. input validation ---------------------------------------------------------
    try:
        encode(4096)
        check("out-of-range message rejected", False)
    except ValueError:
        check("out-of-range message rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
