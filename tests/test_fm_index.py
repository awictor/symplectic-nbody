"""Tests for the FM-index: count/locate match brute force, LF inverts the BWT, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fm_index import FMIndex, brute_count, brute_locate  # noqa: E402


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
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return (state >> 8) / (1 << 24)
    return nxt


def _all_substrings(text, maxlen):
    subs = set()
    n = len(text)
    for i in range(n):
        for L in range(1, maxlen + 1):
            if i + L <= n:
                subs.add(text[i:i + L])
    return subs


def main():
    # ---- 1. count matches brute force over every short substring (structured text) -----
    text = "mississippi"
    fm = FMIndex(text)
    ok = True
    for sub in _all_substrings(text, 5):
        if fm.count(sub) != brute_count(text, sub):
            ok = False
            check("count == brute", False, f"{sub!r}: {fm.count(sub)} vs {brute_count(text, sub)}")
            break
    if ok:
        check("count == brute over all substrings (mississippi)", True)

    # ---- 2. locate returns exactly the correct positions --------------------------------
    ok = True
    for sub in _all_substrings(text, 5):
        if fm.locate(sub) != brute_locate(text, sub):
            ok = False
            check("locate == brute", False, f"{sub!r}")
            break
    if ok:
        check("locate == brute over all substrings", True)

    # ---- 3. specific known counts -------------------------------------------------------
    check("count('issi') = 2", fm.count("issi") == 2, f"{fm.count('issi')}")
    check("count('ss') = 2", fm.count("ss") == 2)
    check("count('i') = 4", fm.count("i") == 4)
    check("locate('issi') = [1,4]", fm.locate("issi") == [1, 4], f"{fm.locate('issi')}")

    # ---- 4. LF-mapping inverts the BWT back to the text ---------------------------------
    check("recover_text == text", fm.recover_text() == text, f"{fm.recover_text()!r}")

    # ---- 5. absent / empty / whole-text patterns ----------------------------------------
    check("absent pattern -> 0", fm.count("xyz") == 0)
    check("absent pattern -> []", fm.locate("xyz") == [])
    check("empty pattern count = len", fm.count("") == len(text))
    check("whole text count = 1", fm.count(text) == 1)
    check("whole text locate = [0]", fm.locate(text) == [0])

    # ---- 6. DNA-like text, random patterns ----------------------------------------------
    rng = _lcg(2024)
    dna = "".join("ACGT"[int(rng() * 4)] for _ in range(200))
    fmd = FMIndex(dna)
    ok = True
    for _ in range(60):
        m = 1 + int(rng() * 6)
        start = int(rng() * (len(dna) - m))
        pat = dna[start:start + m]
        if fmd.count(pat) != brute_count(dna, pat) or fmd.locate(pat) != brute_locate(dna, pat):
            ok = False
            check("DNA count/locate == brute", False, f"{pat!r}")
            break
    if ok:
        check("DNA count/locate == brute (60 patterns)", True)
    check("DNA recover_text == dna", fmd.recover_text() == dna)

    # ---- 7. highly repetitive text (stresses BWT runs and Occ) --------------------------
    rep = "abcabcabcabcabcabc"
    fmr = FMIndex(rep)
    check("repetitive count('abc')", fmr.count("abc") == brute_count(rep, "abc"),
          f"{fmr.count('abc')} vs {brute_count(rep, 'abc')}")
    check("repetitive locate('bca')", fmr.locate("bca") == brute_locate(rep, "bca"))
    check("repetitive recover", fmr.recover_text() == rep)

    # ---- 8. different sampling / checkpoint parameters still correct ---------------------
    for sa_s, occ_s in [(1, 1), (2, 3), (8, 16)]:
        fmx = FMIndex("abracadabra", sa_sample=sa_s, occ_step=occ_s)
        good = (fmx.count("abra") == 2 and fmx.locate("abra") == [0, 7]
                and fmx.recover_text() == "abracadabra")
        check(f"params sa={sa_s},occ={occ_s} correct", good)

    # ---- 9. single-character and unit texts ---------------------------------------------
    fm1 = FMIndex("a")
    check("single char count", fm1.count("a") == 1 and fm1.locate("a") == [0])
    check("single char recover", fm1.recover_text() == "a")

    # ---- 10. sentinel guard -------------------------------------------------------------
    try:
        FMIndex("ab\x00cd")
        check("sentinel in text raises", False)
    except ValueError:
        check("sentinel in text raises", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
