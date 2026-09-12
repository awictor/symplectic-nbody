"""Tests for dancing_links: DLX exact cover vs brute subset search + N-queens counts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dancing_links import solve_exact_cover, brute_exact_cover, n_queens

failed = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failed.append(name)


class LCG:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def rand(self):
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s >> 16

    def randint(self, lo, hi):
        return lo + self.rand() % (hi - lo + 1)


def normalise(sols):
    """Solutions as a set of frozensets of row_ids for order-independent comparison."""
    return {frozenset(s) for s in sols}


# --- Knuth's classic example ------------------------------------------------
items = [1, 2, 3, 4, 5, 6, 7]
options = [('A', {1, 4, 7}), ('B', {1, 4}), ('C', {4, 5, 7}),
           ('D', {3, 5, 6}), ('E', {2, 3, 6, 7}), ('F', {2, 7})]
sols = solve_exact_cover(items, options, find_all=True)
check("Knuth example: unique cover {B, D, F}", normalise(sols) == {frozenset(['B', 'D', 'F'])})
check("first-solution mode returns one cover", len(solve_exact_cover(items, options)) == 1)

# --- trivial / edge cases ---------------------------------------------------
check("single item, single option covers it", solve_exact_cover([1], [('X', {1})]) == [['X']])
check("no options -> no cover for a non-empty item set", solve_exact_cover([1], []) == [])
check("two disjoint options cover two items",
      normalise(solve_exact_cover([1, 2], [('A', {1}), ('B', {2})], find_all=True))
      == {frozenset(['A', 'B'])})
check("overlapping options that cannot tile -> no solution",
      solve_exact_cover([1, 2, 3], [('A', {1, 2}), ('B', {2, 3})], find_all=True) == [])

# --- exhaustive validation vs brute ----------------------------------------
rng = LCG(2026)
match_ok = True
saw_solvable = saw_unsolvable = False
for _ in range(400):
    n_items = rng.randint(1, 7)
    item_list = list(range(n_items))
    n_opts = rng.randint(1, 10)
    options = []
    for k in range(n_opts):
        size = rng.randint(1, n_items)
        s = set()
        while len(s) < size:
            s.add(rng.randint(0, n_items - 1))
        options.append((k, s))
    dlx = normalise(solve_exact_cover(item_list, options, find_all=True))
    brute = normalise(brute_exact_cover(item_list, options, find_all=True))
    if dlx != brute:
        match_ok = False
        print(f"  mismatch: items={item_list} options={options}\n    dlx={dlx}\n    brute={brute}")
        break
    if dlx:
        saw_solvable = True
    else:
        saw_unsolvable = True
check("DLX exact-cover solutions match brute force (400 random instances)", match_ok)
check("suite saw both solvable and unsolvable instances", saw_solvable and saw_unsolvable)

# --- every returned cover is valid -----------------------------------------
rng = LCG(4242)
valid_ok = True
for _ in range(300):
    n_items = rng.randint(1, 8)
    item_list = list(range(n_items))
    options = []
    for k in range(rng.randint(1, 12)):
        size = rng.randint(1, n_items)
        s = set()
        while len(s) < size:
            s.add(rng.randint(0, n_items - 1))
        options.append((k, s))
    opt_map = {k: s for k, s in options}
    for sol in solve_exact_cover(item_list, options, find_all=True):
        # the chosen options must partition the item set exactly
        covered = []
        for rid in sol:
            covered.extend(opt_map[rid])
        if sorted(covered) != item_list:
            valid_ok = False
            break
    if not valid_ok:
        break
check("every returned cover partitions the items exactly once", valid_ok)

# --- N-queens counts --------------------------------------------------------
known = [1, 0, 0, 2, 10, 4, 40, 92]
check("N-queens solution counts for n=1..8 are correct",
      [n_queens(n) for n in range(1, 9)] == known)

# --- a pentomino-like small tiling -----------------------------------------
# tile a 2x3 board (items = 6 cells) with three 1x2 dominoes; count exact covers
cells = [(r, c) for r in range(2) for c in range(3)]
items = cells
options = []
oid = 0
for r in range(2):
    for c in range(3):
        if c + 1 < 3:
            options.append((oid, {(r, c), (r, c + 1)})); oid += 1
        if r + 1 < 2:
            options.append((oid, {(r, c), (r + 1, c)})); oid += 1
dlx_count = len(solve_exact_cover(items, options, find_all=True))
brute_count = len(brute_exact_cover(items, options, find_all=True))
check("2x3 domino tiling count matches brute", dlx_count == brute_count)
check("2x3 board has exactly 3 domino tilings", dlx_count == 3)

# --- limit caps the number of solutions ------------------------------------
# a board with many tilings: 2x4 dominoes has 5 tilings; limit to 2
cells = [(r, c) for r in range(2) for c in range(4)]
options = []
oid = 0
for r in range(2):
    for c in range(4):
        if c + 1 < 4:
            options.append((oid, {(r, c), (r, c + 1)})); oid += 1
        if r + 1 < 2:
            options.append((oid, {(r, c), (r + 1, c)})); oid += 1
capped = solve_exact_cover(cells, options, find_all=True, limit=2)
check("limit caps the number of solutions returned", len(capped) <= 2)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all dancing_links tests passed")
