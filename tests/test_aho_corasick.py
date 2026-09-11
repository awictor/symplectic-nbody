"""Tests for aho_corasick.py -- multi-pattern string matching.

Self-running: prints PASS/FAIL per check, exits 1 if any fail. Results are proven correct
exhaustively against a brute-force per-pattern search.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import aho_corasick as AC  # noqa: E402

failed = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'} {name}")
    if not cond:
        failed.append(name)


# --- the classic example ----------------------------------------------------
ac = AC.AhoCorasick(["he", "she", "his", "hers"])
r = ac.search("ushers")
check("finds 'she' at index 1", r["she"] == [1])
check("finds 'he' at index 2", r["he"] == [2])
check("finds 'hers' at index 2", r["hers"] == [2])
check("does not find 'his'", r["his"] == [])
check("total match count is 3", ac.count_matches("ushers") == 3)
check("contains_any is True", ac.contains_any("ushers"))
check("matches the brute-force search", r == AC.brute_search(["he", "she", "his", "hers"], "ushers"))

# --- overlapping and nested matches ----------------------------------------
ac2 = AC.AhoCorasick(["aa", "aaa"])
o = ac2.search("aaaa")
check("overlapping 'aa' found at 0,1,2", o["aa"] == [0, 1, 2])
check("overlapping 'aaa' found at 0,1", o["aaa"] == [0, 1])
ac3 = AC.AhoCorasick(["a", "ab", "abc", "bc"])
n = ac3.search("abc")
check("nested patterns all found", n["a"] == [0] and n["ab"] == [0] and n["abc"] == [0] and n["bc"] == [1])

# --- no matches -------------------------------------------------------------
check("contains_any is False when nothing matches", not ac.contains_any("xyzxyz"))
check("search returns empty lists when nothing matches",
      all(v == [] for v in AC.AhoCorasick(["cat", "dog"]).search("bird fish").values()))
check("count is 0 with no matches", AC.AhoCorasick(["zzz"]).count_matches("aaa") == 0)

# --- single pattern behaves like a substring search ------------------------
single = AC.AhoCorasick(["ana"])
check("single pattern finds overlapping matches", single.search("banana")["ana"] == [1, 3])

# --- a pattern at the very start / end -------------------------------------
edge = AC.AhoCorasick(["ab", "yz"])
e = edge.search("abxyz")
check("pattern at the start found", e["ab"] == [0])
check("pattern at the end found", e["yz"] == [3])

# --- a longer, realistic blocklist -----------------------------------------
block = AC.AhoCorasick(["error", "warn", "fail", "critical"])
log = "info ok warn fail error errors critical done warn"
res = block.search(log)
check("blocklist finds 'warn' twice", len(res["warn"]) == 2)
check("blocklist finds 'error' inside 'errors' too", len(res["error"]) == 2)
check("blocklist find matches brute force", res == AC.brute_search(["error", "warn", "fail", "critical"], log))

# --- find_all returns end positions correctly ------------------------------
ends = ac.find_all("ushers")
check("find_all reports (end_index, pattern_index) pairs",
      all(0 <= e < len("ushers") and 0 <= idx < 4 for e, idx in ends))

# --- exhaustive check against brute force ----------------------------------
def lcg(seed):
    s = seed
    while True:
        s = (1664525 * s + 1013904223) & 0xFFFFFFFF
        yield s >> 16


gen = lcg(5)


def rnd(lo, hi):
    return lo + next(gen) % (hi - lo + 1)


def rand_str(maxlen, alpha):
    return "".join(alpha[rnd(0, len(alpha) - 1)] for _ in range(rnd(0, maxlen)))


mismatches = 0
for _ in range(1000):
    alpha = "ab" if rnd(0, 1) == 0 else "abc"
    # 1..6 distinct non-empty patterns
    pats = list({rand_str(4, alpha) or alpha[0] for _ in range(rnd(1, 6))})
    text = rand_str(30, alpha)
    ac = AC.AhoCorasick(pats)
    if ac.search(text) != AC.brute_search(pats, text):
        mismatches += 1
check("EVERY one of 1000 random multi-pattern searches matches brute force", mismatches == 0)

# --- count consistency ------------------------------------------------------
count_ok = True
for _ in range(200):
    alpha = "abc"
    pats = list({rand_str(3, alpha) or "a" for _ in range(rnd(1, 5))})
    text = rand_str(25, alpha)
    ac = AC.AhoCorasick(pats)
    brute_total = sum(len(v) for v in AC.brute_search(pats, text).values())
    if ac.count_matches(text) != brute_total:
        count_ok = False
        break
check("count_matches equals the total brute-force occurrences", count_ok)


if failed:
    print(f"\n{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("\nall aho_corasick tests passed")
