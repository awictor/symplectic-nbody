"""Tests for ternary_search_tree: exact map behaviour, prefix, wildcard, deletion vs brute force."""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ternary_search_tree import TernarySearchTree

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


# --- seeded RNG ------------------------------------------------------------
state = 99
def rng():
    global state
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return (state >> 16) / 65536.0


def rand_word():
    n = 1 + int(rng() * 6)
    return "".join(chr(ord("a") + int(rng() * 6)) for _ in range(n))


# --- behaves like a dict ---------------------------------------------------
words = ["cat", "car", "card", "dog", "do", "dodge", "cats", "a", "abc", "ab"]
tst = TernarySearchTree()
ref = {}
for i, w in enumerate(words):
    tst.insert(w, i)
    ref[w] = i

check("size equals number of distinct keys", len(tst) == len(set(words)))
check("all keys present", all(w in tst for w in words))
check("values retrieved correctly", all(tst.get(w) == ref[w] for w in words))
check("absent key not contained", "zzz" not in tst)
check("get default for absent key", tst.get("zzz", -1) == -1)

# --- sorted iteration ------------------------------------------------------
check("keys() is sorted and complete", tst.keys() == sorted(set(words)))
check("items() matches the reference dict", dict(tst.items()) == ref)

# --- re-insert same key updates value, not size ----------------------------
tst.insert("cat", 999)
check("re-insert updates value", tst.get("cat") == 999)
check("re-insert doesn't grow size", len(tst) == len(set(words)))

# --- prefix completion vs brute force --------------------------------------
for pref in ["ca", "car", "do", "d", "a", "cat", "z", ""]:
    got = sorted(tst.keys_with_prefix(pref))
    want = sorted(w for w in set(words) if w.startswith(pref))
    check(f"prefix '{pref}' completion matches brute", got == want)

# --- longest prefix of ----------------------------------------------------
check("longest_prefix_of 'cards' is 'card'", tst.longest_prefix_of("cards") == "card")
check("longest_prefix_of 'doghouse' is 'dog'", tst.longest_prefix_of("doghouse") == "dog")
check("longest_prefix_of 'do' is 'do'", tst.longest_prefix_of("do") == "do")
check("longest_prefix_of 'xyz' is ''", tst.longest_prefix_of("xyz") == "")

# --- wildcard search vs brute-force regex ----------------------------------
for pat in ["c.t", "c..", "...", "d.", "..", "ca.d", ".", "a..", "d.dge"]:
    got = sorted(tst.wildcard(pat))
    rx = re.compile("^" + pat.replace(".", "[a-z]") + "$")
    want = sorted(w for w in set(words) if len(w) == len(pat) and rx.match(w))
    check(f"wildcard '{pat}' matches brute regex", got == want)

# --- deletion behaves like dict deletion -----------------------------------
tst.delete("car")
del ref["car"]
check("deleted key is gone", "car" not in tst)
check("delete shrinks size", len(tst) == len(ref))
check("keys still sorted+correct after delete", tst.keys() == sorted(ref))
check("deleting absent key returns False", tst.delete("nope") is False)
check("prefix 'car' still returns card+cards-family after deleting 'car'",
      sorted(tst.keys_with_prefix("car")) == sorted(w for w in ref if w.startswith("car")))

# --- big randomized cross-check against a dict -----------------------------
big = TernarySearchTree()
bigref = {}
for _ in range(400):
    w = rand_word()
    v = int(rng() * 1000)
    big.insert(w, v)
    bigref[w] = v

check("randomized: size matches dict", len(big) == len(bigref))
check("randomized: keys sorted and complete", big.keys() == sorted(bigref))
check("randomized: all values match", all(big.get(w) == bigref[w] for w in bigref))

# randomized prefix completion
ok_pref = True
for _ in range(60):
    p = rand_word()[: 1 + int(rng() * 2)]
    got = sorted(big.keys_with_prefix(p))
    want = sorted(w for w in bigref if w.startswith(p))
    if got != want:
        ok_pref = False
        break
check("randomized: prefix completion matches brute over 60 queries", ok_pref)

# randomized wildcard
ok_wc = True
for _ in range(80):
    base = rand_word()
    # turn some positions into wildcards
    pat = "".join("." if rng() < 0.4 else ch for ch in base)
    got = sorted(big.wildcard(pat))
    rx = re.compile("^" + pat.replace(".", "[a-z]") + "$")
    want = sorted(w for w in bigref if len(w) == len(pat) and rx.match(w))
    if got != want:
        ok_wc = False
        break
check("randomized: wildcard matches brute regex over 80 queries", ok_wc)

# randomized deletion
dels = list(bigref)[::3]
for w in dels:
    big.delete(w)
    del bigref[w]
check("randomized: keys correct after bulk delete", big.keys() == sorted(bigref))
check("randomized: size correct after bulk delete", len(big) == len(bigref))

# --- empty-key rejected ----------------------------------------------------
try:
    TernarySearchTree().insert("")
    check("empty key rejected", False)
except ValueError:
    check("empty key rejected", True)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all ternary_search_tree tests passed")
