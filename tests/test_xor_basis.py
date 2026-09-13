"""Tests for xor_basis: max/min/count/membership/kth vs brute force, rank, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xor_basis import (XorBasis, brute_reachable, brute_max_xor,  # noqa: E402
                       brute_min_nonempty_xor)


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


def build(nums):
    b = XorBasis()
    for x in nums:
        b.insert(x)
    return b


def main():
    # ---- 1. hand example --------------------------------------------------------------
    b = build([3, 10, 5])          # 3=011, 10=1010, 5=101
    check("max xor of {3,10,5}", b.max_xor() == brute_max_xor([3, 10, 5]))
    check("rank counts independent vectors", b.rank() == 3)
    check("count = 2^rank", b.count() == 8)

    # ---- 2. everything against brute force over random small bags ---------------------
    rng = LCG(2024)
    max_bad = count_bad = min_bad = kth_bad = mem_bad = 0
    for _ in range(500):
        n = rng.randint(0, 9)
        nums = [rng.randint(0, 127) for _ in range(n)]
        b = build(nums)
        reach = brute_reachable(nums)
        if b.max_xor() != brute_max_xor(nums):
            max_bad += 1
        if b.count() != len(reach):
            count_bad += 1
        if nums and b.min_xor() != brute_min_nonempty_xor(nums):
            min_bad += 1
        if [b.kth_smallest(k) for k in range(b.count())] != sorted(reach):
            kth_bad += 1
        # membership: reachable values are contained, a random non-reachable is not
        for x in reach:
            if not b.contains(x):
                mem_bad += 1
        for _ in range(3):
            y = rng.randint(0, 255)
            if b.contains(y) != (y in reach):
                mem_bad += 1
    check("max_xor matches brute force (500 bags)", max_bad == 0, f"{max_bad}")
    check("count = 2^rank matches reachable size", count_bad == 0, f"{count_bad}")
    check("min nonempty xor matches brute force", min_bad == 0, f"{min_bad}")
    check("kth_smallest enumerates sorted reachable set", kth_bad == 0, f"{kth_bad}")
    check("membership matches brute force", mem_bad == 0, f"{mem_bad}")

    # ---- 3. rank behaviour ------------------------------------------------------------
    b = XorBasis()
    grew = [b.insert(v) for v in [1, 2, 3]]   # 3 = 1 XOR 2, so third insert is dependent
    check("independent inserts grow, dependent does not", grew == [True, True, False])
    check("rank is 2 for {1,2,3}", b.rank() == 2)
    check("dependent insert leaves span unchanged", b.count() == 4)
    check("3 is in the span of {1,2}", b.contains(3))
    check("4 is not in the span of {1,2}", not b.contains(4))

    # ---- 4. min_xor is 0 when a dependency exists -------------------------------------
    b = build([1, 2, 3])          # 1^2^3 = 0 -> nonempty subset XORs to 0
    check("min nonempty xor is 0 given a dependency", b.min_xor() == 0)
    b = build([1, 2, 4])          # independent
    check("min nonempty xor is smallest pivot when independent", b.min_xor() == 1)

    # ---- 5. max_xor with a starting value ---------------------------------------------
    b = build([1, 2, 4])
    # starting from 8, can reach 8..15
    check("max_xor from a start value", b.max_xor(8) == 15)

    # ---- 6. edge cases ----------------------------------------------------------------
    empty = XorBasis()
    check("empty basis reaches only 0", empty.max_xor() == 0 and empty.count() == 1)
    check("empty basis min_xor is None", empty.min_xor() is None)
    check("empty basis contains 0, not 5", empty.contains(0) and not empty.contains(5))
    z = build([0, 0, 0])
    check("all-zero inserts stay rank 0", z.rank() == 0 and z.count() == 1)
    check("min nonempty xor of zeros is 0", z.min_xor() == 0)
    try:
        build([1, 2]).kth_smallest(4)
        check("kth out of range raises", False)
    except IndexError:
        check("kth out of range raises", True)

    # ---- 7. large bag: max beats any sampled subset -----------------------------------
    rng = LCG(99)
    nums = [rng.randint(0, (1 << 40) - 1) for _ in range(200)]
    b = build(nums)
    mx = b.max_xor()
    beaten = True
    for _ in range(2000):
        v = 0
        for x in nums:
            if rng.randint(0, 1):
                v ^= x
        if v > mx:
            beaten = False
    check("large-bag max_xor beats 2000 random subsets", beaten)
    check("large-bag rank <= 40 (bit width)", b.rank() <= 40)

    # ---- 8. arbitrary width (Python big ints) -----------------------------------------
    big = [1 << 100, (1 << 100) | 1, 1 << 200]
    b = build(big)
    check("handles 200-bit values", b.max_xor() == (1 << 200) | (1 << 100) | 1)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
