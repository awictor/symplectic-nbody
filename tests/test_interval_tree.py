"""Tests for interval_tree: exact agreement with brute force on stab/overlap, invariants, edge cases."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from interval_tree import IntervalTree, brute_stab, brute_overlap  # noqa: E402


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


def as_set(triples):
    return sorted((lo, hi) for lo, hi, _ in triples)


def check_invariants(node):
    """Every stored interval contains the center; lists sorted; partition correct."""
    if node is None:
        return True
    for lo, hi, _ in node.by_left:
        if not (lo <= node.center <= hi):
            return False
    # by_left sorted by lo, by_right sorted by hi desc
    if any(node.by_left[i][0] > node.by_left[i + 1][0] for i in range(len(node.by_left) - 1)):
        return False
    if any(node.by_right[i][1] < node.by_right[i + 1][1] for i in range(len(node.by_right) - 1)):
        return False
    return check_invariants(node.left) and check_invariants(node.right)


def main():
    # ---- 1. hand example --------------------------------------------------------------
    ivals = [(1, 5), (3, 8), (6, 10), (12, 15), (0, 2)]
    t = IntervalTree(ivals)
    check("stab 4 matches brute", as_set(t.stab(4)) == as_set(brute_stab(ivals, 4)))
    check("stab 7 matches brute", as_set(t.stab(7)) == as_set(brute_stab(ivals, 7)))
    check("stab 11 (gap) is empty", t.stab(11) == [])
    check("overlap [4,7] matches brute", as_set(t.overlap(4, 7)) == as_set(brute_overlap(ivals, 4, 7)))
    check("tree invariants hold", check_invariants(t.root))
    check("len is interval count", len(t) == 5)

    # ---- 2. exact agreement with brute force over random sets -------------------------
    rng = LCG(2024)
    stab_bad = 0
    ov_bad = 0
    for trial in range(40):
        n = rng.randint(1, 120)
        ivals = []
        for _ in range(n):
            lo = rng.randint(0, 200)
            hi = lo + rng.randint(0, 40)
            ivals.append((lo, hi))
        t = IntervalTree(ivals)
        if not check_invariants(t.root):
            check(f"invariants (trial {trial})", False)
            break
        for _ in range(40):
            q = rng.randint(-10, 210)
            if as_set(t.stab(q)) != as_set(brute_stab(ivals, q)):
                stab_bad += 1
        for _ in range(40):
            a = rng.randint(-10, 210)
            b = a + rng.randint(0, 60)
            if as_set(t.overlap(a, b)) != as_set(brute_overlap(ivals, a, b)):
                ov_bad += 1
    check("stabbing queries match brute force (1600 queries)", stab_bad == 0, f"{stab_bad} mismatches")
    check("range overlap queries match brute force (1600 queries)", ov_bad == 0, f"{ov_bad} mismatches")

    # ---- 3. counts equal reported list lengths ----------------------------------------
    ivals = [(rng.randint(0, 100), 0) for _ in range(50)]
    ivals = [(lo, lo + rng.randint(0, 30)) for lo, _ in ivals]
    t = IntervalTree(ivals)
    q = 50
    check("count_stab equals len(stab)", t.count_stab(q) == len(t.stab(q)))
    check("count_overlap equals len(overlap)", t.count_overlap(20, 60) == len(t.overlap(20, 60)))

    # ---- 4. degenerate: zero-length intervals (points) --------------------------------
    pts = [(5, 5), (5, 5), (10, 10), (7, 7)]
    t = IntervalTree(pts)
    check("stab a point interval", as_set(t.stab(5)) == as_set(brute_stab(pts, 5)))
    check("stab misses between points", t.stab(6) == [])

    # ---- 5. touching endpoints (closed intervals) ------------------------------------
    ivals = [(1, 3), (3, 5), (5, 7)]
    t = IntervalTree(ivals)
    # point 3 is contained by both [1,3] and [3,5]
    check("shared endpoint hits both closed intervals", len(t.stab(3)) == 2)
    check("overlap [3,3] hits both", as_set(t.overlap(3, 3)) == as_set(brute_overlap(ivals, 3, 3)))

    # ---- 6. nested and identical intervals -------------------------------------------
    ivals = [(0, 100), (10, 90), (40, 60), (49, 51), (49, 51)]
    t = IntervalTree(ivals)
    check("nested intervals all found at center", len(t.stab(50)) == 5)
    check("identical intervals both reported", as_set(t.stab(50)) == as_set(brute_stab(ivals, 50)))

    # ---- 7. queries outside the whole range -------------------------------------------
    ivals = [(10, 20), (15, 25), (30, 40)]
    t = IntervalTree(ivals)
    check("stab far left empty", t.stab(-100) == [])
    check("stab far right empty", t.stab(1000) == [])
    check("overlap far range empty", t.overlap(500, 600) == [])
    check("overlap covering everything returns all", len(t.overlap(0, 1000)) == 3)

    # ---- 8. data payloads preserved ---------------------------------------------------
    ivals = [(1, 5, "a"), (3, 8, "b"), (6, 10, "c")]
    t = IntervalTree(ivals)
    hits = t.stab(4)
    labels = sorted(d for _, _, d in hits)
    check("data payloads preserved", labels == ["a", "b"])

    # ---- 9. empty tree and validation -------------------------------------------------
    empty = IntervalTree([])
    check("empty tree stab is empty", empty.stab(5) == [])
    check("empty tree overlap is empty", empty.overlap(0, 10) == [])
    try:
        IntervalTree([(5, 3)])
        check("lo > hi rejected", False)
    except ValueError:
        check("lo > hi rejected", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
