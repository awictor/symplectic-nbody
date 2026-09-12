"""Tests for degree_sequence: Havel-Hakimi / Erdos-Gallai realizability vs brute enumeration."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from degree_sequence import (is_graphic_erdos_gallai, is_graphic_havel_hakimi, realize,
                             reduction_steps, degree_sequence, brute_is_graphic)

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


def is_simple_with_degrees(n, edges, seq):
    """The edges form a simple graph (no loops/multi-edges) with exactly the degree sequence seq."""
    seen = set()
    for u, v in edges:
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    return degree_sequence(n, edges) == sorted(seq, reverse=True)


# --- known cases ------------------------------------------------------------
check("(3,3,3,1) is not graphic", not is_graphic_erdos_gallai([3, 3, 3, 1]))
check("(2,2,2) triangle is graphic", is_graphic_erdos_gallai([2, 2, 2]))
check("(3,3,2,2) is graphic", is_graphic_erdos_gallai([3, 3, 2, 2]))
check("(1,1,1) has odd sum -> not graphic", not is_graphic_erdos_gallai([1, 1, 1]))
check("(3,3,3,3) = K4 is graphic", is_graphic_erdos_gallai([3, 3, 3, 3]))
check("all-zero sequence is graphic (edgeless)", is_graphic_erdos_gallai([0, 0, 0]))
check("empty sequence is graphic", is_graphic_erdos_gallai([]))
check("a degree exceeding n-1 is not graphic", not is_graphic_erdos_gallai([5, 1, 1, 1]))
check("(4,1,1,1,1) is graphic (a star K_{1,4})", is_graphic_erdos_gallai([4, 1, 1, 1, 1]))
check("Havel-Hakimi agrees on (3,3,3,1)", not is_graphic_havel_hakimi([3, 3, 3, 1]))

# --- Havel-Hakimi and Erdos-Gallai always agree, and match brute -----------
rng = LCG(2026)
agree_ok = brute_ok = True
saw_graphic = saw_not = False
for _ in range(600):
    n = rng.randint(1, 7)
    seq = [rng.randint(0, n - 1) for _ in range(n)]
    eg = is_graphic_erdos_gallai(seq)
    hh = is_graphic_havel_hakimi(seq)
    if eg != hh:
        agree_ok = False
        print(f"  EG/HH disagree on {seq}: EG={eg} HH={hh}")
        break
    if eg != brute_is_graphic(seq):
        brute_ok = False
        print(f"  vs brute mismatch on {seq}: criteria={eg} brute={brute_is_graphic(seq)}")
        break
    if eg:
        saw_graphic = True
    else:
        saw_not = True
check("Erdos-Gallai and Havel-Hakimi always agree (600 sequences)", agree_ok)
check("both criteria match brute-force realizability (600 sequences)", brute_ok)
check("suite saw both graphic and non-graphic sequences", saw_graphic and saw_not)

# --- realize() builds a simple graph with the exact degrees ----------------
rng = LCG(4242)
realize_ok = True
for _ in range(400):
    n = rng.randint(1, 8)
    seq = [rng.randint(0, n - 1) for _ in range(n)]
    g = realize(seq)
    if is_graphic_erdos_gallai(seq):
        if g is None or not is_simple_with_degrees(n, g, seq):
            realize_ok = False
            print(f"  realize failed for graphic {seq}: {g}")
            break
    else:
        if g is not None:
            realize_ok = False
            break
check("realize() builds a simple graph with the exact degree sequence, else None (400 seqs)",
      realize_ok)

# --- realizing a KNOWN graph's degree sequence round-trips -----------------
rng = LCG(777)
roundtrip_ok = True
for _ in range(300):
    n = rng.randint(2, 8)
    # random simple graph
    edges = []
    seen = set()
    for _ in range(rng.randint(0, n * 2)):
        u = rng.randint(0, n - 1)
        v = rng.randint(0, n - 1)
        if u != v and (min(u, v), max(u, v)) not in seen:
            seen.add((min(u, v), max(u, v)))
            edges.append((u, v))
    seq = degree_sequence(n, edges)
    # its degree sequence must be graphic, and realize() must reproduce those degrees
    if not is_graphic_erdos_gallai(seq):
        roundtrip_ok = False
        break
    g = realize(seq)
    if not is_simple_with_degrees(n, g, seq):
        roundtrip_ok = False
        break
check("a real graph's degree sequence is graphic and re-realizable (300 graphs)", roundtrip_ok)

# --- reduction steps end in all-zeros for graphic sequences ----------------
steps, ok = reduction_steps([3, 3, 2, 2])
check("Havel-Hakimi reduction of (3,3,2,2) succeeds", ok)
check("its reduction ends at all-zeros", steps[-1] == [] or all(x == 0 for x in steps[-1]))
_, ok = reduction_steps([3, 3, 3, 1])
check("Havel-Hakimi reduction of (3,3,3,1) reports failure", not ok)

# --- regular sequences: d-regular on n vertices is graphic iff n*d even ----
reg_ok = True
for n in range(2, 9):
    for d in range(0, n):
        seq = [d] * n
        expected = (n * d) % 2 == 0        # and d <= n-1, always true here
        if is_graphic_erdos_gallai(seq) != expected:
            reg_ok = False
            break
    if not reg_ok:
        break
check("d-regular sequence on n vertices is graphic iff n*d is even", reg_ok)

print()
if failed:
    print(f"{len(failed)} FAILED: {failed}")
    sys.exit(1)
print("all degree_sequence tests passed")
