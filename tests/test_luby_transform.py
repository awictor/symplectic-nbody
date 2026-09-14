"""Tests for LT codes: soliton distributions, encode/peel round-trip, erasure robustness, graceful failure."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import luby_transform as LT  # noqa: E402


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
    # ---- 1. soliton distributions are valid probability distributions -------------------
    for k in (5, 20, 100):
        ide = LT.ideal_soliton(k)
        rob = LT.robust_soliton(k)
        check(f"ideal soliton (k={k}) sums to 1", abs(sum(ide) - 1.0) < 1e-9, f"{sum(ide)}")
        check(f"robust soliton (k={k}) sums to 1", abs(sum(rob) - 1.0) < 1e-9, f"{sum(rob)}")
        check(f"robust soliton (k={k}) non-negative", all(p >= 0 for p in rob))

    # ---- 2. chunking splits and pads correctly ------------------------------------------
    data = bytes(range(37))
    blocks, orig = LT.chunk_message(data, 8)
    check("chunk length correct", orig == 37)
    check("all blocks equal length", all(len(b) == 8 for b in blocks))
    check("chunks reconstruct the original", b"".join(blocks)[:orig] == data)

    # ---- 3. a degree-1 symbol trivially decodes its block -------------------------------
    b0 = bytes([1, 2, 3, 4])
    b1 = bytes([5, 6, 7, 8])
    syms = [((0,), b0), ((1,), b1), ((0, 1), bytes(a ^ b for a, b in zip(b0, b1)))]
    dec = LT.peel_decode(syms, 2, 4)
    check("degree-1 symbols decode directly", dec == [b0, b1])

    # ---- 4. full message round-trip at 2x overhead --------------------------------------
    msg = b"The quick brown fox jumps over the lazy dog. " * 12
    blen = 16
    blocks, orig = LT.chunk_message(msg, blen)
    k = len(blocks)
    syms = LT.LTEncoder(blocks, seed=3).generate(int(k * 2.0) + 5)
    recovered = LT.decode(syms, k, blen)
    check("message recovered exactly at 2x overhead", recovered is not None and recovered[:orig] == msg)

    # ---- 5. decoded blocks match the originals exactly ----------------------------------
    blocks_dec = LT.peel_decode(syms, k, blen)
    check("every decoded block matches", blocks_dec is not None and
          all(blocks_dec[i] == blocks[i] for i in range(k)))

    # ---- 6. re-encoding a decoded message reproduces the symbols (XOR self-consistency) --
    # a symbol's payload equals the XOR of the decoded blocks it references
    ok = True
    for nb, pl in syms[:20]:
        x = bytearray(blen)
        for idx in nb:
            for i in range(blen):
                x[i] ^= blocks_dec[idx][i]
        if bytes(x) != pl:
            ok = False
    check("symbols are XORs of their decoded blocks", ok)

    # ---- 7. erasure robustness: any sufficient random subset decodes --------------------
    rnd = _lcg(9)
    big = LT.LTEncoder(blocks, seed=11).generate(k * 3)
    # keep a random ~2.2k subset
    keep = [s for s in big if rnd() < 0.75]
    dec_e = LT.peel_decode(keep, k, blen)
    check("random surviving subset decodes", dec_e is not None and
          all(dec_e[i] == blocks[i] for i in range(k)), f"kept {len(keep)} ({len(keep)/k:.2f}k)")

    # ---- 8. graceful failure when too few symbols --------------------------------------
    few = LT.LTEncoder(blocks, seed=5).generate(k // 2)
    check("too few symbols -> None", LT.peel_decode(few, k, blen) is None)

    # ---- 9. encoder/decoder agree via the shared seed (determinism) ---------------------
    s1 = LT.LTEncoder(blocks, seed=42).generate(10)
    s2 = LT.LTEncoder(blocks, seed=42).generate(10)
    check("same seed -> identical symbols",
          all(s1[i][0] == s2[i][0] and s1[i][1] == s2[i][1] for i in range(10)))
    s3 = LT.LTEncoder(blocks, seed=43).generate(10)
    check("different seed -> different symbols", any(s1[i][0] != s3[i][0] for i in range(10)))

    # ---- 10. a single-block message is trivially handled --------------------------------
    one, o1 = LT.chunk_message(b"hello world!", 16)
    check("single block chunking", len(one) == 1)
    sy = LT.LTEncoder(one, seed=1).generate(3)
    d1 = LT.peel_decode(sy, 1, 16)
    check("single-block message decodes", d1 is not None and d1[0] == one[0])

    # ---- 11. larger message decodes at 1.5x (bigger k -> lower overhead) ----------------
    big_msg = bytes((i * 37 + 11) % 256 for i in range(4000))
    bb, ob = LT.chunk_message(big_msg, 20)
    kk = len(bb)
    ss = LT.LTEncoder(bb, seed=7).generate(int(kk * 1.5))
    rec = LT.decode(ss, kk, 20)
    check("larger message (k=200) decodes at 1.5x overhead",
          rec is not None and rec[:ob] == big_msg, f"k={kk}")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
