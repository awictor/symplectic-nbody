"""Tests for myers_diff: distance == N+M-2*LCS vs DP, apply reproduces B, LCS is a subsequence."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from myers_diff import (edit_distance, edit_script, apply_script,  # noqa: E402
                        longest_common_subsequence, unified_diff, lcs_length_dp,
                        KEEP, DELETE, INSERT)


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


def is_subsequence(sub, seq):
    it = iter(seq)
    return all(x in it for x in sub)


def main():
    rng = LCG(2024)
    alpha = "abcd"

    # ---- 1. hand examples --------------------------------------------------------------
    check("identical strings -> distance 0", edit_distance("abcabba", "abcabba") == 0)
    check("empty vs empty -> 0", edit_distance("", "") == 0)
    check("empty vs nonempty -> len", edit_distance("", "abc") == 3)
    check("classic ABCABBA->CBABAC distance", edit_distance("ABCABBA", "CBABAC") == 5)

    # ---- 2. distance == N + M - 2*LCS, cross-checked with DP over random pairs ---------
    mism = 0
    for _ in range(400):
        la = rng.randint(12)
        lb = rng.randint(12)
        a = "".join(alpha[rng.randint(len(alpha))] for _ in range(la))
        b = "".join(alpha[rng.randint(len(alpha))] for _ in range(lb))
        d = edit_distance(a, b)
        lcs = lcs_length_dp(a, b)
        expected = len(a) + len(b) - 2 * lcs
        if d != expected:
            mism += 1
            if mism <= 5:
                print(f"    MISMATCH a={a!r} b={b!r} d={d} expected={expected} lcs={lcs}")
    check("edit distance == N+M-2*LCS over 400 random pairs", mism == 0, f"{mism} mismatches")

    # ---- 3. applying the script reproduces B (correctness, not just length) ------------
    bad = 0
    for _ in range(400):
        la = rng.randint(14)
        lb = rng.randint(14)
        a = "".join(alpha[rng.randint(len(alpha))] for _ in range(la))
        b = "".join(alpha[rng.randint(len(alpha))] for _ in range(lb))
        script = edit_script(a, b)
        rebuilt = apply_script(a, script)
        if rebuilt != list(b):
            bad += 1
            if bad <= 5:
                print(f"    APPLY FAIL a={a!r} b={b!r} got={''.join(rebuilt)!r}")
    check("apply_script(a, diff(a,b)) == b over 400 pairs", bad == 0, f"{bad} failures")

    # ---- 4. script length equals the edit distance ------------------------------------
    lenbad = 0
    for _ in range(200):
        a = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        b = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        script = edit_script(a, b)
        nedits = sum(1 for op, _ in script if op != KEEP)
        if nedits != edit_distance(a, b):
            lenbad += 1
    check("script edit count == edit_distance", lenbad == 0, f"{lenbad} mismatches")

    # ---- 5. LCS is a genuine subsequence of both, with LCS length ----------------------
    subbad = 0
    for _ in range(200):
        a = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        b = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        lcs = longest_common_subsequence(a, b)
        if not (is_subsequence(lcs, a) and is_subsequence(lcs, b) and len(lcs) == lcs_length_dp(a, b)):
            subbad += 1
    check("LCS is a subsequence of both with correct length", subbad == 0, f"{subbad} failures")

    # ---- 6. symmetry of distance ------------------------------------------------------
    symbad = 0
    for _ in range(200):
        a = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        b = "".join(alpha[rng.randint(len(alpha))] for _ in range(rng.randint(12)))
        if edit_distance(a, b) != edit_distance(b, a):
            symbad += 1
    check("edit distance symmetric", symbad == 0, f"{symbad} mismatches")

    # ---- 7. disjoint inputs -> delete all + insert all --------------------------------
    d = edit_distance("aaa", "bbb")
    check("disjoint inputs distance == len(a)+len(b)", d == 6, f"{d}")
    script = edit_script("aaa", "bbb")
    check("disjoint script is 3 deletes + 3 inserts",
          sum(1 for op, _ in script if op == DELETE) == 3
          and sum(1 for op, _ in script if op == INSERT) == 3)

    # ---- 8. line-based diff (git-style) ------------------------------------------------
    a_lines = ["import os", "import sys", "", "def main():", "    return 1"]
    b_lines = ["import os", "import json", "", "def main():", "    return 0"]
    diff = unified_diff(a_lines, b_lines)
    # applying the char/line script reproduces b_lines
    check("line diff applies back to b", apply_script(a_lines, edit_script(a_lines, b_lines)) == b_lines)
    # unchanged lines are kept
    check("unified diff keeps common lines", " import os" in diff and " def main():" in diff)
    check("unified diff marks changes", any(l.startswith("-import sys") for l in diff)
          and any(l.startswith("+import json") for l in diff))

    # ---- 9. long mostly-similar inputs (Myers' sweet spot) ----------------------------
    base = [str(i) for i in range(500)]
    modified = base[:250] + ["XXX"] + base[251:]   # one line changed
    d = edit_distance(base, modified)
    check("single change in 500 lines -> distance 2", d == 2, f"{d}")
    check("large diff applies back", apply_script(base, edit_script(base, modified)) == modified)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
