"""Tests for thompson_nfa: agreement with Python re over random patterns, no catastrophic backtracking."""

import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from thompson_nfa import Regex, fullmatch, search, parse, compile_nfa  # noqa: E402


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


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def nxt(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s

    def randint(self, n):
        return (self.nxt() >> 8) % n

    def choice(self, seq):
        return seq[self.randint(len(seq))]


def random_regex(rng, alphabet, depth=0):
    """Generate a random regex from the supported grammar (no escapes, to keep re-equivalence clean)."""
    if depth >= 3:
        # base case: a literal or dot
        return rng.choice(alphabet + ['.'])
    r = rng.randint(10)
    if r < 4:
        return rng.choice(alphabet + ['.'])
    if r < 6:
        # concatenation
        return random_regex(rng, alphabet, depth + 1) + random_regex(rng, alphabet, depth + 1)
    if r < 8:
        # quantifier
        sub = random_regex(rng, alphabet, depth + 1)
        # wrap multi-char subs in a group so the quantifier binds sensibly
        if len(sub) > 1:
            sub = '(' + sub + ')'
        return sub + rng.choice(['*', '+', '?'])
    if r < 9:
        # alternation
        return ('(' + random_regex(rng, alphabet, depth + 1) + '|'
                + random_regex(rng, alphabet, depth + 1) + ')')
    # group
    return '(' + random_regex(rng, alphabet, depth + 1) + ')'


def random_string(rng, alphabet, maxlen=6):
    n = rng.randint(maxlen + 1)
    return ''.join(rng.choice(alphabet) for _ in range(n))


def main():
    # ---- 1. hand-written examples -----------------------------------------------------
    cases = [
        ("abc", "abc", True), ("abc", "abd", False),
        ("a*", "", True), ("a*", "aaaa", True), ("a*", "aaab", False),
        ("a+", "", False), ("a+", "a", True), ("a+", "aaa", True),
        ("a?b", "b", True), ("a?b", "ab", True), ("a?b", "aab", False),
        ("a|b", "a", True), ("a|b", "b", True), ("a|b", "c", False),
        ("(ab)*", "", True), ("(ab)*", "ababab", True), ("(ab)*", "aba", False),
        ("a.c", "abc", True), ("a.c", "axc", True), ("a.c", "ac", False),
        ("(a|b)*c", "ababbac", True), ("(a|b)*c", "ababba", False),
        ("colou?r", "color", True), ("colou?r", "colour", True), ("colou?r", "colur", False),
    ]
    for pat, text, expect in cases:
        got = fullmatch(pat, text)
        check(f"fullmatch {pat!r} vs {text!r} == {expect}", got == expect, f"got {got}")

    # ---- 2. escaped metacharacters ----------------------------------------------------
    check("escaped star literal", fullmatch(r"a\*b", "a*b") and not fullmatch(r"a\*b", "aaab"))
    check("escaped dot literal", fullmatch(r"a\.c", "a.c") and not fullmatch(r"a\.c", "axc"))

    # ---- 3. agreement with Python re over random patterns (fullmatch) -----------------
    rng = LCG(12345)
    alphabet = ['a', 'b', 'c']
    mism = 0
    trials = 500
    for _ in range(trials):
        pat = random_regex(rng, alphabet)
        txt = random_string(rng, alphabet)
        try:
            ref = re.fullmatch(pat, txt) is not None
        except re.error:
            continue
        got = fullmatch(pat, txt)
        if got != ref:
            mism += 1
            if mism <= 5:
                print(f"    MISMATCH fullmatch pat={pat!r} txt={txt!r} got={got} re={ref}")
    check(f"fullmatch agrees with re over {trials} random cases", mism == 0, f"{mism} mismatches")

    # ---- 4. agreement with Python re for search --------------------------------------
    rng2 = LCG(99)
    mism_s = 0
    for _ in range(trials):
        pat = random_regex(rng2, alphabet)
        txt = random_string(rng2, alphabet, maxlen=8)
        try:
            ref = re.search(pat, txt) is not None
        except re.error:
            continue
        got = search(pat, txt)
        if got != ref:
            mism_s += 1
            if mism_s <= 5:
                print(f"    MISMATCH search pat={pat!r} txt={txt!r} got={got} re={ref}")
    check(f"search agrees with re over {trials} random cases", mism_s == 0, f"{mism_s} mismatches")

    # ---- 5. no catastrophic backtracking ---------------------------------------------
    # (a*)*b against a long run of 'a' (no trailing b) -> a backtracking engine explodes; Thompson
    # returns promptly.
    rx = Regex("(a*)*b")
    t0 = time.time()
    res = rx.fullmatch("a" * 200)
    elapsed = time.time() - t0
    check("catastrophic pattern returns quickly", elapsed < 1.0, f"took {elapsed:.3f}s")
    check("catastrophic pattern correctly rejects (no trailing b)", res is False)
    check("catastrophic pattern accepts with trailing b", Regex("(a*)*b").fullmatch("a" * 200 + "b"))

    # ---- 6. linear scaling sanity: doubling input roughly doubles time, not squares ----
    def timed(n):
        r = Regex("(a|aa)*")
        t0 = time.time()
        r.fullmatch("a" * n)
        return time.time() - t0
    t100 = timed(100)
    t400 = timed(400)
    # 4x input should be far under 16x time (quadratic) -- allow generous slack for noise
    check("scaling is sub-quadratic", t400 < max(t100 * 12, 0.5), f"t100={t100:.4f} t400={t400:.4f}")

    # ---- 7. empty pattern and epsilon -------------------------------------------------
    check("empty pattern matches empty string", fullmatch("", ""))
    check("empty pattern rejects nonempty", not fullmatch("", "a"))

    # ---- 8. parse errors ---------------------------------------------------------------
    for bad in ["(a", "a)", "*a", "a|*"]:
        try:
            parse(bad)
            check(f"parse rejects {bad!r}", False)
        except ValueError:
            check(f"parse rejects {bad!r}", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
