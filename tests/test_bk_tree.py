"""Tests for bk_tree: exact agreement with brute force, pruning fires, nearest match, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bk_tree import BKTree, brute_search, levenshtein  # noqa: E402


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

    def randint(self, lo, hi):
        return lo + (self.nxt() >> 8) % (hi - lo + 1)

    def word(self, alpha, maxlen):
        n = self.randint(1, maxlen)
        return "".join(alpha[self.randint(0, len(alpha) - 1)] for _ in range(n))


def main():
    # ---- 1. hand example --------------------------------------------------------------
    words = ["book", "books", "boo", "boon", "cook", "cake", "cape", "cart"]
    bt = BKTree()
    bt.add_all(words)
    check("size matches distinct words", len(bt) == len(set(words)))
    got = bt.search("book", 1)
    ref = brute_search(words, "book", 1)
    check("search 'book' tol 1 matches brute force", got == ref, f"{got} vs {ref}")
    check("all words recoverable", set(bt.words()) == set(words))

    # ---- 2. exact agreement with brute force over random dictionaries -----------------
    rng = LCG(2024)
    alpha = "abcde"
    mism = 0
    total_visited = 0
    total_n = 0
    for trial in range(30):
        dict_words = list({rng.word(alpha, 7) for _ in range(200)})
        bt = BKTree()
        bt.add_all(dict_words)
        for _ in range(50):
            q = rng.word(alpha, 7)
            tol = rng.randint(0, 3)
            got = bt.search(q, tol)
            ref = brute_search(dict_words, q, tol)
            if got != ref:
                mism += 1
            total_visited += bt.last_visited
            total_n += len(dict_words)
    check("BK-tree search matches brute force (1500 queries)", mism == 0, f"{mism} mismatches")

    # ---- 3. pruning fires: far fewer nodes visited than brute force -------------------
    avg_frac = total_visited / total_n
    check("pruning visits far fewer than all nodes", avg_frac < 0.7,
          f"visited {100*avg_frac:.0f}% of nodes on average")

    # ---- 4. nearest match equals the true minimum ------------------------------------
    rng = LCG(7)
    dict_words = list({rng.word(alpha, 8) for _ in range(300)})
    bt = BKTree()
    bt.add_all(dict_words)
    near_bad = 0
    for _ in range(200):
        q = rng.word(alpha, 8)
        d, w = bt.nearest(q)
        true_min = min(levenshtein(q, x) for x in dict_words)
        if d != true_min:
            near_bad += 1
    check("nearest match is the true minimum distance", near_bad == 0, f"{near_bad} wrong")

    # ---- 5. exact match at distance 0 -------------------------------------------------
    bt = BKTree()
    bt.add_all(["apple", "apply", "ample"])
    check("exact query returns distance 0", bt.search("apple", 0) == [(0, "apple")])
    check("nearest of an exact member is itself", bt.nearest("apply") == (0, "apply"))

    # ---- 6. tolerance larger than any distance returns everything ---------------------
    got = bt.search("apple", 100)
    check("huge tolerance returns all words", set(w for _, w in got) == {"apple", "apply", "ample"})

    # ---- 7. insertion order doesn't change results -----------------------------------
    words = ["cat", "car", "cot", "cut", "bat", "bar", "can", "ban", "scat", "scar"]
    rng = LCG(99)
    for _ in range(5):
        shuffled = words[:]
        for i in range(len(shuffled) - 1, 0, -1):
            j = rng.randint(0, i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        bt = BKTree()
        bt.add_all(shuffled)
        got = bt.search("cat", 1)
        ref = brute_search(words, "cat", 1)
        if got != ref:
            check("insertion order invariant", False, f"{got} vs {ref}")
            break
    else:
        check("insertion order does not change results", True)

    # ---- 8. edge cases ----------------------------------------------------------------
    empty = BKTree()
    check("empty tree search is empty", empty.search("x", 3) == [])
    check("empty tree nearest is None", empty.nearest("x") is None)
    check("empty tree size 0", len(empty) == 0)
    bt = BKTree()
    bt.add("solo")
    bt.add("solo")           # duplicate ignored
    check("duplicate insert ignored", len(bt) == 1)

    # ---- 9. real-ish word list ---------------------------------------------------------
    real = ["algorithm", "logarithm", "rhythm", "altruism", "alignment", "argument",
            "arithmetic", "allocator", "alligator", "alternator", "gorilla", "algae"]
    bt = BKTree()
    bt.add_all(real)
    got = bt.search("algorithn", 2)      # typo of algorithm
    ref = brute_search(real, "algorithn", 2)
    check("real word list fuzzy search matches brute force", got == ref, f"{got}")
    check("nearest to a typo is the intended word", bt.nearest("algorithn")[1] == "algorithm")

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
