"""Tests for cuckoo_hash: mirror dict over random ops, two-probe guarantee, invariants, load factor."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cuckoo_hash import CuckooHash  # noqa: E402


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


def probes_at_most_two(ch, key):
    """Verify the key is found by inspecting only its two hash cells."""
    i1 = ch._h1(key)
    i2 = ch._h2(key)
    in1 = ch.t1[i1] is not None and ch.t1[i1][0] == key
    in2 = ch.t2[i2] is not None and ch.t2[i2][0] == key
    return in1 or in2


def main():
    # ---- 1. basic insert / lookup / delete -------------------------------------------
    ch = CuckooHash()
    for k in range(20):
        ch.insert(k, k * 10)
    check("all inserted keys found", all(ch.get(k) == k * 10 for k in range(20)))
    check("absent key returns default", ch.get(999) is None and ch.get(999, -1) == -1)
    check("membership works", 5 in ch and 999 not in ch)
    ch.delete(5)
    check("delete removes key", 5 not in ch and len(ch) == 19)
    check("invariants after basic ops", ch.check_invariants())

    # ---- 2. mirror a dict over thousands of random operations ------------------------
    rng = LCG(2024)
    ch = CuckooHash(seed=7)
    ref = {}
    agree = True
    inv_ok = True
    two_probe = True
    for step in range(5000):
        op = rng.randint(0, 3)
        key = rng.randint(0, 300)
        if op <= 1:                          # insert / overwrite
            val = rng.nxt()
            ch.insert(key, val)
            ref[key] = val
        elif op == 2:                        # delete
            r = ch.delete(key)
            if r != (key in ref):
                agree = False
            ref.pop(key, None)
        else:                                # lookup + membership
            if ch.get(key) != ref.get(key):
                agree = False
            if (key in ch) != (key in ref):
                agree = False
        # every present key must be found within its two cells
        if op == 3 and key in ref and not probes_at_most_two(ch, key):
            two_probe = False
        if not ch.check_invariants():
            inv_ok = False
            break
    check("agrees with dict over 5000 random ops", agree)
    check("invariants hold throughout", inv_ok)
    check("size matches dict", len(ch) == len(ref))
    check("keys match dict", sorted(ch.keys()) == sorted(ref.keys()))

    # ---- 3. the worst-case guarantee: every key at one of its two cells ---------------
    all_two = all(probes_at_most_two(ch, k) for k in ref)
    check("every key found in at most two cells", all_two)
    check("max probe cost is 2", ch.max_probe_cost() == 2)

    # ---- 4. load factor stays bounded ------------------------------------------------
    check("load factor below the configured max", ch.load_factor() <= 0.45 + 1e-9,
          f"{ch.load_factor():.3f}")

    # ---- 5. overwrite doesn't change size --------------------------------------------
    ch2 = CuckooHash()
    ch2.insert(42, "a")
    ch2.insert(42, "b")
    check("overwrite updates value", ch2.get(42) == "b")
    check("overwrite keeps size 1", len(ch2) == 1)

    # ---- 6. growth preserves contents ------------------------------------------------
    ch3 = CuckooHash(capacity=4, seed=3)
    data = {i: i * i for i in range(200)}
    for k, v in data.items():
        ch3.insert(k, v)
    check("all keys survive repeated growth/rehash", all(ch3.get(k) == v for k, v in data.items()))
    check("size correct after growth", len(ch3) == 200)
    check("invariants after growth", ch3.check_invariants())

    # ---- 7. delete frees space for reinsertion ---------------------------------------
    ch4 = CuckooHash(capacity=8, seed=9)
    for k in range(6):
        ch4.insert(k, k)
    ch4.delete(3)
    ch4.insert(100, 100)          # should slot into freed space
    check("reinsert after delete works", ch4.get(100) == 100 and 3 not in ch4)

    # ---- 8. string keys ---------------------------------------------------------------
    ch5 = CuckooHash(seed=11)
    words = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
    for i, w in enumerate(words):
        ch5.insert(w, i)
    check("string keys work", all(ch5.get(w) == i for i, w in enumerate(words)))
    check("string key invariants", ch5.check_invariants())

    # ---- 9. iteration returns exactly the stored keys --------------------------------
    ch6 = CuckooHash(seed=5)
    ks = set(rng.randint(0, 1000) for _ in range(100))
    for k in ks:
        ch6.insert(k, k)
    check("iteration returns exactly the keys", set(ch6.keys()) == ks and len(ch6.keys()) == len(ks))

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
